#!/usr/bin/env python3
"""
Satellite Strategy Backtest
===========================

A low-correlation, capital-defending "satellite" overlay designed to sit next to a
long-term buy-and-hold core.  Dual momentum (absolute + relative) across broad,
liquid asset-class ETFs, gated by a trend-regime filter, sized by realised
volatility, protected by ATR trailing stops, and validated with a strict
out-of-sample architecture.

Pipeline
--------
STAGE A  Data ingestion & cleaning       -> fetch_prices(), align_universe()
STAGE B  Systematic indicators           -> build_indicators()
STAGE C  Anti-overfitting engine         -> holdout_validation(), rolling_walk_forward()
STAGE D  Analytical engine & reporting   -> performance_table(), crisis_table(), main()

Strategy (decision at close t, execution at close t+1, weekly cadence by default)
--------------------------------------------------------------------------------
1. Absolute momentum   : asset L-day total return minus cash-proxy L-day return > 0
2. Trend filter        : asset close > SMA(W) of its own close
3. Relative momentum   : rank eligible assets by excess momentum, hold top-K
4. Volatility sizing   : w_i = vol_target / (K * sigma_i), sigma_i = 20d realised vol
                         (a no-trade band suppresses tiny weekly re-sizing trades)
5. Regime cash buffer  : SPY > SMA(W)                  -> gross risk cap 100 %
                         SPY < SMA(W), SPY abs-mom > 0 -> gross risk cap `bear_cap`
                         SPY < SMA(W), SPY abs-mom < 0 -> gross risk cap `crisis_cap`
6. Trailing stop       : exit a position when close < (peak close since entry - k * ATR14)
7. Idle capital earns the cash-proxy return (BIL, spliced with SHY before BIL's listing).

Anti-overfitting
----------------
* Parameters are chosen ONLY on the first 70 % of history (train); the final 30 %
  is a pure out-of-sample test run once with the frozen parameters.
* Selection uses a neighbourhood-smoothed objective (plateau search, not spike search).
* The Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) is reported for the
  in-sample winner, correcting for the number of trials in the grid.
* The Probabilistic Sharpe Ratio is reported out-of-sample (skew/kurtosis adjusted).
* Optional `--wfo` runs a rolling walk-forward (re-optimise every `step` years) and
  stitches the OOS folds into one continuous equity curve.

Usage
-----
    python satellite_strategy_backtest.py                 # 70/30 holdout, live Yahoo data
    python satellite_strategy_backtest.py --wfo           # + rolling walk-forward
    python satellite_strategy_backtest.py --synthetic     # offline smoke test (fake data!)
    python satellite_strategy_backtest.py --fast          # reduced grid, quick run
    python satellite_strategy_backtest.py --rebalance M --objective sortino --cost-bps 10
    python satellite_strategy_backtest.py --help

Outputs (in --output-dir, default ./satellite_output): summary_oos.csv, grid_results_train.csv,
oos_equity_curves.csv, oos_weights.csv, wfo_folds.csv / wfo_equity_curves.csv (with --wfo),
oos_equity.png (if matplotlib is installed).

Dependencies: pandas, numpy, scipy, yfinance  (matplotlib optional, for the PNG chart)
"""
from __future__ import annotations

import argparse
import itertools
import logging
import math
import sys
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:  # yfinance is only needed for live data; --synthetic works without it
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

LOG = logging.getLogger("satellite")
TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

RISK_ASSETS: Tuple[str, ...] = ("SPY", "TLT", "GLD")   # equities, long bonds, gold
CASH_PROXIES: Tuple[str, ...] = ("BIL", "SHY")          # BIL preferred; SHY fills pre-2007
BENCHMARK = "SPY"

CRISIS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "GFC 2007-09": ("2007-10-09", "2009-03-09"),
    "Flash/EU 2011": ("2011-07-22", "2011-10-03"),
    "Q4 2018": ("2018-09-20", "2018-12-24"),
    "COVID 2020": ("2020-02-19", "2020-03-23"),
    "Inflation 2022": ("2022-01-03", "2022-10-12"),
    "Tariff shock 2025": ("2025-02-19", "2025-04-08"),
}


@dataclass(frozen=True)
class StrategyParams:
    """All knobs of the satellite strategy.  Frozen so it can be a dict key."""

    mom_lookback: int = 126       # momentum lookback (trading days)
    sma_window: int = 200         # trend / regime filter window (trading days)
    top_k: int = 2                # max simultaneous risk positions
    vol_target: float = 0.10      # annualised portfolio volatility target
    stop_atr_mult: float = 3.0    # trailing stop distance in ATR units (<= 0 disables)
    bear_cap: float = 0.75        # gross risk cap when SPY < SMA but SPY abs-mom > 0
    crisis_cap: float = 0.0       # gross risk cap when SPY < SMA and SPY abs-mom < 0
    max_weight: float = 1.0       # per-asset weight cap
    rebalance_band: float = 0.05  # no-trade band: ignore |target - current| below this (held names only)
    vol_window: int = 20          # realised-vol window for sizing
    atr_window: int = 14          # ATR window for stops

    def label(self) -> str:
        return (f"L={self.mom_lookback} SMA={self.sma_window} K={self.top_k} "
                f"stop={self.stop_atr_mult:g}ATR crisis_cap={self.crisis_cap:g}")


def default_grid(fast: bool = False) -> List[StrategyParams]:
    """Parameter grid searched on the TRAIN segment only."""
    if fast:
        mom = (126, 252)
        sma = (150, 200)
        top_k = (1, 2)
        stops = (0.0, 3.0)
        crisis = (0.0, 0.5)
    else:
        mom = (63, 126, 189, 252)
        sma = (100, 150, 200, 250)
        top_k = (1, 2)
        stops = (0.0, 3.0)
        crisis = (0.0, 0.5)
    grid = [StrategyParams(mom_lookback=m, sma_window=s, top_k=k,
                           stop_atr_mult=st, crisis_cap=c)
            for m, s, k, st, c in itertools.product(mom, sma, top_k, stops, crisis)]
    return grid


# ----------------------------------------------------------------------------
# STAGE A - DATA INGESTION & CLEANING
# ----------------------------------------------------------------------------

def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker}.csv"


def _normalise_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean DataFrame with columns Open/High/Low/Close (float, tz-naive index)."""
    cols = {c.lower(): c for c in df.columns}
    needed = {}
    for want in ("open", "high", "low", "close"):
        if want not in cols:
            raise ValueError(f"missing column {want!r}")
        needed[want.capitalize()] = df[cols[want]]
    out = pd.DataFrame(needed).astype(float)
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Close"])
    return out


def _download_yf(tickers: Sequence[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Batch download via yfinance with per-ticker fallback.  Never raises."""
    out: Dict[str, pd.DataFrame] = {}
    if yf is None:
        LOG.error("yfinance is not installed: pip install yfinance")
        return out
    try:
        raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                          progress=False, threads=True, group_by="column")
    except Exception as exc:  # network / API failure
        LOG.warning("batch download failed (%s); falling back to per-ticker", exc)
        raw = pd.DataFrame()
    for t in tickers:
        df = pd.DataFrame()
        try:
            if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(1):
                df = raw.xs(t, axis=1, level=1)
            elif not isinstance(raw.columns, pd.MultiIndex) and len(tickers) == 1:
                df = raw
            if df.empty or df["Close"].dropna().empty:
                LOG.info("per-ticker fallback for %s", t)
                df = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
            df = _normalise_ohlc(df)
            if df.empty:
                raise ValueError("empty frame")
            out[t] = df
        except Exception as exc:
            LOG.warning("could not fetch %s: %s", t, exc)
    return out


def fetch_prices(tickers: Sequence[str], start: str, end: str, cache_dir: Path,
                 use_cache: bool = True, max_stale_days: int = 5) -> Dict[str, pd.DataFrame]:
    """STAGE A.1 - fetch adjusted OHLC per ticker, with a local CSV cache.

    Missing / failed tickers are logged and omitted (never raise).  `auto_adjust=True`
    gives dividend- and split-adjusted prices, i.e. a total-return proxy, which is
    essential for TLT / BIL where most of the return is coupon income.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    to_fetch: List[str] = []
    end_ts = pd.Timestamp(end)
    for t in tickers:
        p = _cache_path(cache_dir, t)
        if use_cache and p.exists():
            try:
                df = _normalise_ohlc(pd.read_csv(p, index_col=0))
                age = (end_ts - df.index[-1]).days
                if age <= max_stale_days:
                    out[t] = df
                    LOG.info("cache hit  %-4s %s -> %s", t, df.index[0].date(), df.index[-1].date())
                    continue
                LOG.info("cache stale %-4s (%d days); refetching", t, age)
            except Exception as exc:
                LOG.warning("bad cache for %s (%s); refetching", t, exc)
        to_fetch.append(t)
    if to_fetch:
        fresh = _download_yf(to_fetch, start, end)
        for t, df in fresh.items():
            out[t] = df
            try:
                df.to_csv(_cache_path(cache_dir, t))
            except OSError as exc:
                LOG.warning("could not write cache for %s: %s", t, exc)
            LOG.info("downloaded %-4s %s -> %s (%d rows)", t, df.index[0].date(),
                     df.index[-1].date(), len(df))
    return out


def synthetic_prices(start: str, end: str, seed: int = 7) -> Dict[str, pd.DataFrame]:
    """Regime-switching correlated GBM with OHLC - for OFFLINE SMOKE TESTS ONLY.

    Two-state Markov chain (bull / bear) drives SPY drift & vol; TLT, GLD, cash
    are correlated with SPY via a fixed Cholesky factor.  BIL is blanked before
    2007-05-30 to exercise the BIL/SHY splice logic.  Results from this data say
    nothing about real markets.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    T = len(idx)
    # Markov regime: 0 = bull, 1 = bear
    p_stay = np.array([[0.995, 0.005], [0.97, 0.03]])  # row: current, col: next
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        regime[t] = 0 if rng.random() < p_stay[regime[t - 1], 0] else 1
    # Annualised drift / vol per asset per regime (SPY, TLT, GLD, BIL, SHY)
    mu = np.array([[0.13, 0.04, 0.07, 0.015, 0.02],
                   [-0.35, 0.12, 0.05, 0.015, 0.03]])
    sg = np.array([[0.13, 0.14, 0.15, 0.003, 0.012],
                   [0.38, 0.20, 0.22, 0.003, 0.012]])
    corr = np.array([[1.0, -0.30, 0.05, 0.0, -0.05],
                     [-0.30, 1.0, 0.15, 0.0, 0.70],
                     [0.05, 0.15, 1.0, 0.0, 0.10],
                     [0.0, 0.0, 0.0, 1.0, 0.0],
                     [-0.05, 0.70, 0.10, 0.0, 1.0]])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((T, 5)) @ L.T
    dt = 1.0 / TRADING_DAYS
    r = mu[regime] * dt + sg[regime] * np.sqrt(dt) * z - 0.5 * (sg[regime] ** 2) * dt
    close = 100.0 * np.exp(np.cumsum(r, axis=0))
    names = ["SPY", "TLT", "GLD", "BIL", "SHY"]
    out = {}
    for j, name in enumerate(names):
        c = close[:, j]
        o = np.r_[c[0], c[:-1]] * (1 + 0.002 * rng.standard_normal(T) * (sg[regime, j] / 0.15))
        wiggle = np.abs(rng.standard_normal(T)) * sg[regime, j] * np.sqrt(dt) * 0.8
        h = np.maximum(o, c) * (1 + wiggle)
        lo = np.minimum(o, c) * (1 - wiggle)
        df = pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c}, index=idx)
        if name == "BIL":
            df = df[df.index >= "2007-05-30"]
        out[name] = df
    return out


@dataclass
class Universe:
    """Aligned market data as numpy arrays (T x N) plus metadata."""

    dates: pd.DatetimeIndex
    tickers: List[str]
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    ret: np.ndarray        # simple daily returns of risk assets
    cash_ret: np.ndarray   # simple daily return of the cash proxy
    spy_idx: int
    cash_label: str


def align_universe(prices: Dict[str, pd.DataFrame], risk_assets: Sequence[str],
                   cash_proxies: Sequence[str], benchmark: str) -> Universe:
    """STAGE A.2 - align timestamps, forward-fill, splice the cash proxy.

    * Calendar = trading days of the benchmark (SPY).
    * Risk assets: outer-join then forward-fill (count of filled cells is logged);
      history starts at the first date on which EVERY risk asset has a price.
    * Cash proxy: BIL daily returns, with SHY returns filling dates before BIL
      existed, and 0 % where neither exists.
    """
    missing = [t for t in risk_assets if t not in prices]
    if missing:
        raise RuntimeError(f"required risk assets unavailable: {missing}. "
                           "Check tickers / network, or run with --synthetic.")
    if benchmark not in prices:
        raise RuntimeError(f"benchmark {benchmark} unavailable")
    calendar = prices[benchmark].index
    fields = {}
    for f in ("Close", "High", "Low"):
        frame = pd.concat({t: prices[t][f] for t in risk_assets}, axis=1).reindex(calendar)
        n_nan_before = int(frame.isna().sum().sum())
        frame = frame.ffill()
        filled = n_nan_before - int(frame.isna().sum().sum())
        if filled:
            LOG.info("forward-filled %d %s cells", filled, f)
        fields[f] = frame
    first_valid = max(fields["Close"][t].first_valid_index() for t in risk_assets)
    for f in fields:
        fields[f] = fields[f].loc[first_valid:]
    dates = fields["Close"].index
    # --- cash proxy splice ---------------------------------------------------
    cash = pd.Series(0.0, index=dates)
    have = np.zeros(len(dates), dtype=bool)
    used = []
    for t in cash_proxies:
        if t not in prices:
            LOG.warning("cash proxy %s unavailable; skipping", t)
            continue
        r = prices[t]["Close"].reindex(dates).ffill().pct_change(fill_method=None)
        fill = (~have) & r.notna().to_numpy()
        cash[fill] = r[fill]
        have |= fill
        used.append(f"{t}:{int(fill.sum())}d")
    if not have.any():
        LOG.warning("no cash proxy available; idle capital earns 0 %")
    cash_label = " + ".join(used) if used else "none (0%)"
    LOG.info("cash proxy splice -> %s", cash_label)
    close = fields["Close"].to_numpy(dtype=float)
    ret = np.vstack([np.zeros((1, close.shape[1])), close[1:] / close[:-1] - 1.0])
    return Universe(dates=dates, tickers=list(risk_assets), close=close,
                    high=fields["High"].to_numpy(dtype=float),
                    low=fields["Low"].to_numpy(dtype=float), ret=ret,
                    cash_ret=cash.to_numpy(dtype=float),
                    spy_idx=list(risk_assets).index(benchmark), cash_label=cash_label)


# ----------------------------------------------------------------------------
# STAGE B - SYSTEMATIC INDICATOR GENERATION
# ----------------------------------------------------------------------------

@dataclass
class Indicators:
    sma: Dict[int, np.ndarray]        # window -> (T x N)
    mom: Dict[int, np.ndarray]        # lookback -> (T x N) total return over L days
    cash_mom: Dict[int, np.ndarray]   # lookback -> (T,) cash total return over L days
    vol: Dict[int, np.ndarray]        # window -> (T x N) annualised realised vol
    atr: Dict[int, np.ndarray]        # window -> (T x N) Wilder ATR in price units


def build_indicators(u: Universe, grid: Iterable[StrategyParams]) -> Indicators:
    """Pre-compute every indicator window the grid will ever ask for (once).

    All indicators at row t use data up to and including close t - no look-ahead.
    """
    grid = list(grid)
    close = pd.DataFrame(u.close, index=u.dates, columns=u.tickers)
    high = pd.DataFrame(u.high, index=u.dates, columns=u.tickers)
    low = pd.DataFrame(u.low, index=u.dates, columns=u.tickers)
    logret = np.log(close).diff()
    cash_log = pd.Series(np.log1p(u.cash_ret), index=u.dates)

    sma = {w: close.rolling(w).mean().to_numpy() for w in {p.sma_window for p in grid}}
    mom, cash_mom = {}, {}
    for L in {p.mom_lookback for p in grid}:
        mom[L] = (close / close.shift(L) - 1.0).to_numpy()
        cash_mom[L] = np.expm1(cash_log.rolling(L).sum()).to_numpy()
    vol = {w: (logret.rolling(w).std() * math.sqrt(TRADING_DAYS)).to_numpy()
           for w in {p.vol_window for p in grid}}
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                   axis=1, keys=["hl", "hc", "lc"]).T.groupby(level=1).max().T
    tr = tr[u.tickers]
    atr = {w: tr.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean().to_numpy()
           for w in {p.atr_window for p in grid}}
    return Indicators(sma=sma, mom=mom, cash_mom=cash_mom, vol=vol, atr=atr)


def rebalance_mask(dates: pd.DatetimeIndex, freq: str = "W") -> np.ndarray:
    """True on decision days: last trading day of each week ("W"), month ("M") or every day ("D")."""
    if freq.upper() == "D":
        return np.ones(len(dates), dtype=bool)
    if freq.upper() == "W":
        iso = dates.isocalendar()
        key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    elif freq.upper() == "M":
        key = pd.Series(dates.year * 100 + dates.month, index=dates)
    else:
        raise ValueError("rebalance freq must be D, W or M")
    key = pd.Series(np.asarray(key), index=dates)
    return (key != key.shift(-1)).to_numpy()


# ----------------------------------------------------------------------------
# BACKTEST ENGINE  (daily loop; path-dependent stops need it)
# ----------------------------------------------------------------------------

@dataclass
class EngineResult:
    returns: pd.Series          # daily net portfolio returns
    weights: pd.DataFrame       # end-of-day risk weights
    gross: pd.Series            # gross risk exposure
    turnover: float             # total one-way turnover (sum |dw|)
    n_rebalances: int
    n_stops: int


def _target_weights(t: int, p: StrategyParams, u: Universe, ind: Indicators) -> np.ndarray:
    """Dual momentum + trend filter + vol sizing + regime cash buffer, using data <= t."""
    n = len(u.tickers)
    mom_t = ind.mom[p.mom_lookback][t]
    cm = ind.cash_mom[p.mom_lookback][t]
    sma_t = ind.sma[p.sma_window][t]
    vol_t = ind.vol[p.vol_window][t]
    c = u.close[t]
    if not (np.isfinite(mom_t).all() and np.isfinite(sma_t).all()
            and np.isfinite(vol_t).all() and np.isfinite(cm)):
        return np.zeros(n)                       # warm-up: stay in cash
    excess = mom_t - cm                           # absolute momentum vs cash
    # --- market regime (SPY) -> gross cap -------------------------------------
    if c[u.spy_idx] > sma_t[u.spy_idx]:
        cap = 1.0
    elif excess[u.spy_idx] > 0:
        cap = p.bear_cap
    else:
        cap = p.crisis_cap
    # --- eligibility and relative ranking -------------------------------------
    eligible = (excess > 0) & (c > sma_t)
    w = np.zeros(n)
    if cap <= 0 or not eligible.any():
        return w
    order = np.argsort(-excess, kind="stable")
    chosen = [i for i in order if eligible[i]][:p.top_k]
    for i in chosen:                              # inverse-vol sizing to target
        w[i] = min(p.max_weight, p.vol_target / (p.top_k * max(vol_t[i], 1e-4)))
    g = w.sum()
    if g > cap:
        w *= cap / g
    return w


def run_engine(u: Universe, ind: Indicators, p: StrategyParams, start: int, end: int,
               decision_days: np.ndarray, cost_bps: float = 5.0) -> EngineResult:
    """Simulate [start, end) with trade-next-close execution and proportional costs.

    Day-t sequence:
      1. accrue return close(t-1)->close(t) with weights held overnight, drift weights
      2. execute the target decided at t-1 (cost = turnover * cost_bps)
      3. update trailing peaks
      4. if t is a decision day, compute a new target for execution at t+1
      5. check ATR stops on positions held at close t -> exit at close t+1
    """
    n = len(u.tickers)
    atr = ind.atr[p.atr_window]
    cost = cost_bps * 1e-4
    w = np.zeros(n)
    pending: Optional[np.ndarray] = None
    peak = np.full(n, np.nan)
    T = end - start
    port = np.zeros(T)
    gross = np.zeros(T)
    wts = np.zeros((T, n))
    turnover = 0.0
    n_rebal = n_stops = 0
    for k, t in enumerate(range(start, end)):
        r = u.ret[t]
        rp = float(w @ r) + (1.0 - w.sum()) * u.cash_ret[t]
        if rp > -1.0:
            w = w * (1.0 + r) / (1.0 + rp)
        if pending is not None:
            dw = np.abs(pending - w).sum()
            turnover += dw
            rp -= dw * cost
            entered = (pending > 0) & (w <= 0)
            peak[entered] = u.close[t, entered]
            peak[pending <= 0] = np.nan
            w = pending
            pending = None
        held = w > 0
        if held.any():
            peak[held] = np.fmax(peak[held], u.close[t, held])
        port[k] = rp
        gross[k] = w.sum()
        wts[k] = w
        if decision_days[t]:
            target = _target_weights(t, p, u, ind)
            if p.rebalance_band > 0:          # no-trade band on names already held and still wanted
                keep = held & (target > 0) & (np.abs(target - w) <= p.rebalance_band)
                target[keep] = w[keep]
            pending = target
            n_rebal += 1
        if p.stop_atr_mult > 0 and held.any():
            a = atr[t]
            trig = held & np.isfinite(a) & (u.close[t] < peak - p.stop_atr_mult * a)
            if trig.any():
                if pending is None:
                    pending = w.copy()
                pending = pending.copy()
                pending[trig] = 0.0
                n_stops += int(trig.sum())
    idx = u.dates[start:end]
    return EngineResult(returns=pd.Series(port, index=idx, name="satellite"),
                        weights=pd.DataFrame(wts, index=idx, columns=u.tickers),
                        gross=pd.Series(gross, index=idx), turnover=turnover,
                        n_rebalances=n_rebal, n_stops=n_stops)


# ----------------------------------------------------------------------------
# PERFORMANCE STATISTICS
# ----------------------------------------------------------------------------

def _drawdown_stats(equity: pd.Series) -> Dict[str, object]:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough_i = int(np.argmin(dd.to_numpy()))
    mdd = float(dd.iloc[trough_i])
    peak_i = int(np.argmax(equity.to_numpy()[: trough_i + 1]))
    after = equity.to_numpy()[trough_i:]
    rec = np.where(after >= equity.iloc[peak_i])[0]
    if len(rec):
        rec_i = trough_i + int(rec[0])
        recovery_days = rec_i - trough_i
        recovered = True
    else:
        rec_i = len(equity) - 1
        recovery_days = rec_i - trough_i
        recovered = False
    return {"max_drawdown": mdd, "dd_peak": equity.index[peak_i], "dd_trough": equity.index[trough_i],
            "dd_recovery": equity.index[rec_i] if recovered else pd.NaT,
            "recovery_days": recovery_days, "recovered": recovered,
            "peak_to_recovery_days": rec_i - peak_i}


def fast_sharpe(returns: np.ndarray, rf: np.ndarray) -> float:
    x = returns - rf
    s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(TRADING_DAYS)) if s > 0 else 0.0


def fast_objective(returns: np.ndarray, rf: np.ndarray, objective: str) -> float:
    x = returns - rf
    if objective == "sharpe":
        return fast_sharpe(returns, rf)
    if objective == "sortino":
        d = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
        return float(x.mean() / d * math.sqrt(TRADING_DAYS)) if d > 0 else 0.0
    if objective == "calmar":
        eq = np.cumprod(1.0 + returns)
        mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
        cagr = eq[-1] ** (TRADING_DAYS / len(returns)) - 1.0
        return float(cagr / abs(mdd)) if mdd < 0 else 0.0
    raise ValueError(objective)


def probabilistic_sharpe(returns: np.ndarray, rf: np.ndarray, sr_benchmark_daily: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012): P[true SR > benchmark], skew/kurtosis adjusted."""
    x = returns - rf
    T = len(x)
    s = x.std(ddof=1)
    if T < 10 or s <= 0:
        return float("nan")
    sr = x.mean() / s
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark_daily) * math.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe(returns: np.ndarray, rf: np.ndarray, trial_sr_annual: Sequence[float]) -> Tuple[float, float]:
    """DSR (Bailey & Lopez de Prado 2014): PSR against the expected max SR of N trials.

    Returns (DSR, SR*_annualised).  N = number of grid trials (treated as independent,
    which is conservative because neighbouring parameter sets are correlated).
    """
    tr = np.asarray(trial_sr_annual, dtype=float) / math.sqrt(TRADING_DAYS)
    N = len(tr)
    if N < 2:
        return float("nan"), float("nan")
    V = tr.var(ddof=1)
    sr_star = math.sqrt(V) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1.0 / N)
                              + EULER_GAMMA * stats.norm.ppf(1 - 1.0 / (N * math.e)))
    return probabilistic_sharpe(returns, rf, sr_star), sr_star * math.sqrt(TRADING_DAYS)


def performance_stats(returns: pd.Series, rf: pd.Series, bench: Optional[pd.Series] = None) -> Dict[str, object]:
    """STAGE D - full statistic set for one daily return series."""
    r = returns.to_numpy(dtype=float)
    f = rf.reindex(returns.index).fillna(0.0).to_numpy(dtype=float)
    x = r - f
    T = len(r)
    eq = pd.Series(np.cumprod(1.0 + r), index=returns.index)
    years = T / TRADING_DAYS
    total = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sd = x.std(ddof=1)
    sharpe = float(x.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    sharpe_se = float(math.sqrt((1 + 0.5 * (sharpe / math.sqrt(TRADING_DAYS)) ** 2) / T) * math.sqrt(TRADING_DAYS))
    dd_dev = math.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    sortino = float(x.mean() / dd_dev * math.sqrt(TRADING_DAYS)) if dd_dev > 0 else float("nan")
    dd = _drawdown_stats(eq)
    out = {"total_return": total, "cagr": cagr, "ann_vol": vol, "sharpe": sharpe, "sharpe_se": sharpe_se,
           "sortino": sortino, "calmar": cagr / abs(dd["max_drawdown"]) if dd["max_drawdown"] < 0 else float("nan"),
           "psr_vs_0": probabilistic_sharpe(r, f), "skew": float(stats.skew(x)),
           "excess_kurt": float(stats.kurtosis(x)), "worst_day": float(r.min()),
           "best_day": float(r.max()), "pct_positive_days": float((r > 0).mean()), "n_days": T}
    out.update(dd)
    if bench is not None:
        b = bench.reindex(returns.index).to_numpy(dtype=float)
        bx = b - f
        if bx.std() > 0:
            out["corr_vs_spy"] = float(np.corrcoef(x, bx)[0, 1])
            out["beta_vs_spy"] = float(np.cov(x, bx, ddof=1)[0, 1] / bx.var(ddof=1))
            down = bx < 0
            out["downside_capture"] = float(x[down].mean() / bx[down].mean()) if down.any() else float("nan")
            up = bx > 0
            out["upside_capture"] = float(x[up].mean() / bx[up].mean()) if up.any() else float("nan")
    return out


def performance_table(series: Dict[str, pd.Series], rf: pd.Series, bench_key: str) -> pd.DataFrame:
    bench = series[bench_key]
    df = pd.DataFrame({name: performance_stats(s, rf, bench=bench) for name, s in series.items()})
    order = ["total_return", "cagr", "ann_vol", "sharpe", "sharpe_se", "sortino", "calmar",
             "max_drawdown", "dd_peak", "dd_trough", "dd_recovery", "recovery_days", "peak_to_recovery_days",
             "recovered", "psr_vs_0", "corr_vs_spy", "beta_vs_spy", "downside_capture", "upside_capture",
             "skew", "excess_kurt", "worst_day", "best_day", "pct_positive_days", "n_days"]
    return df.reindex([o for o in order if o in df.index])


def format_table(df: pd.DataFrame) -> str:
    pct = {"total_return", "cagr", "ann_vol", "max_drawdown", "worst_day", "best_day",
           "pct_positive_days", "psr_vs_0", "downside_capture", "upside_capture"}
    two = {"sharpe", "sharpe_se", "sortino", "calmar", "corr_vs_spy", "beta_vs_spy", "skew", "excess_kurt"}
    out = df.copy().astype(object)
    for idx in out.index:
        for col in out.columns:
            v = out.at[idx, col]
            if idx in pct and isinstance(v, (float, int, np.floating)):
                out.at[idx, col] = f"{100 * v:,.2f}%"
            elif idx in two and isinstance(v, (float, int, np.floating)):
                out.at[idx, col] = f"{v:,.2f}"
            elif isinstance(v, pd.Timestamp):
                out.at[idx, col] = v.strftime("%Y-%m-%d")
            elif v is pd.NaT:
                out.at[idx, col] = "not yet"
    return out.to_string()


# ----------------------------------------------------------------------------
# BENCHMARKS & BLENDS
# ----------------------------------------------------------------------------

def rebalanced_mix(returns: pd.DataFrame, weights: Sequence[float], freq: str = "M") -> pd.Series:
    """Constant-mix portfolio rebalanced at `freq` boundaries (drifts in between)."""
    w0 = np.asarray(weights, dtype=float)
    r = returns.to_numpy(dtype=float)
    mask = rebalance_mask(returns.index, freq)
    w = w0.copy()
    out = np.zeros(len(r))
    for t in range(len(r)):
        rp = float(w @ r[t])
        out[t] = rp
        w = w * (1 + r[t]) / (1 + rp)
        if mask[t]:
            w = w0.copy()
    return pd.Series(out, index=returns.index)


# ----------------------------------------------------------------------------
# STAGE C - ANTI-OVERFITTING ENGINE
# ----------------------------------------------------------------------------

@dataclass
class GridResult:
    table: pd.DataFrame            # one row per parameter set (train metrics)
    best: StrategyParams
    best_smoothed_score: float


def grid_search(u: Universe, ind: Indicators, grid: Sequence[StrategyParams], start: int, end: int,
                decision_days: np.ndarray, cost_bps: float, objective: str,
                smooth_weight: float = 0.5) -> GridResult:
    """Evaluate every parameter set on [start, end) and pick the most ROBUST one.

    Robustness: the raw objective is blended with the mean objective of its grid
    neighbours along the two continuous dimensions (mom_lookback, sma_window).
    This favours plateaus over isolated spikes - the classic parameter-surface
    heuristic for avoiding over-fitted optima (Pardo, 2008).
    """
    rows = []
    rf = u.cash_ret[start:end]
    t0 = time.time()
    for i, p in enumerate(grid):
        res = run_engine(u, ind, p, start, end, decision_days, cost_bps)
        r = res.returns.to_numpy()
        eq = np.cumprod(1 + r)
        mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
        rows.append({**asdict(p), "objective": fast_objective(r, rf, objective),
                     "sharpe": fast_sharpe(r, rf), "cagr": eq[-1] ** (TRADING_DAYS / len(r)) - 1,
                     "max_drawdown": mdd, "turnover_py": res.turnover / (len(r) / TRADING_DAYS),
                     "n_stops": res.n_stops, "avg_gross": float(res.gross.mean())})
        if (i + 1) % 25 == 0 or i + 1 == len(grid):
            LOG.info("  grid %3d/%d  (%.0fs)", i + 1, len(grid), time.time() - t0)
    tab = pd.DataFrame(rows)
    # --- neighbourhood smoothing ------------------------------------------------
    moms = sorted(tab["mom_lookback"].unique())
    smas = sorted(tab["sma_window"].unique())
    other = [c for c in ("top_k", "stop_atr_mult", "crisis_cap", "vol_target", "bear_cap",
                         "max_weight", "rebalance_band", "vol_window", "atr_window") if c in tab.columns]
    key = tab.set_index(["mom_lookback", "sma_window"] + other)["objective"].to_dict()
    smoothed = []
    for _, row in tab.iterrows():
        mi, si = moms.index(row["mom_lookback"]), smas.index(row["sma_window"])
        rest = tuple(row[c] for c in other)
        neigh = []
        for dm, ds in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            mj, sj = mi + dm, si + ds
            if 0 <= mj < len(moms) and 0 <= sj < len(smas):
                v = key.get((moms[mj], smas[sj]) + rest)
                if v is not None and np.isfinite(v):
                    neigh.append(v)
        own = row["objective"]
        smoothed.append(own if not neigh else (1 - smooth_weight) * own + smooth_weight * np.mean(neigh))
    tab["smoothed_objective"] = smoothed
    tab = tab.sort_values("smoothed_objective", ascending=False).reset_index(drop=True)
    best_row = tab.iloc[0]
    best = replace(grid[0], **{k: (int(best_row[k]) if isinstance(getattr(grid[0], k), int) else float(best_row[k]))
                               for k in asdict(grid[0])})
    return GridResult(table=tab, best=best, best_smoothed_score=float(best_row["smoothed_objective"]))


def holdout_validation(u: Universe, ind: Indicators, grid: Sequence[StrategyParams], decision_days: np.ndarray,
                       cost_bps: float, objective: str, train_frac: float = 0.70) -> Dict[str, object]:
    """Single chronological split: optimise on the first `train_frac`, test once on the rest."""
    T = len(u.dates)
    split = int(T * train_frac)
    LOG.info("STAGE C  holdout split: train %s -> %s (%d d) | test %s -> %s (%d d)",
             u.dates[0].date(), u.dates[split - 1].date(), split,
             u.dates[split].date(), u.dates[-1].date(), T - split)
    gs = grid_search(u, ind, grid, 0, split, decision_days, cost_bps, objective)
    best = gs.best
    LOG.info("  selected (smoothed %s=%.3f): %s", objective, gs.best_smoothed_score, best.label())
    train_res = run_engine(u, ind, best, 0, split, decision_days, cost_bps)
    test_res = run_engine(u, ind, best, split, T, decision_days, cost_bps)
    dsr, sr_star = deflated_sharpe(train_res.returns.to_numpy(), u.cash_ret[:split], gs.table["sharpe"].tolist())
    return {"split": split, "grid": gs, "best": best, "train": train_res, "test": test_res,
            "dsr": dsr, "sr_star": sr_star}


def rolling_walk_forward(u: Universe, ind: Indicators, grid: Sequence[StrategyParams], decision_days: np.ndarray,
                         cost_bps: float, objective: str, train_years: float = 6.0, test_years: float = 2.0,
                         min_warmup_days: int = 260) -> Dict[str, object]:
    """Rolling WFO: optimise on `train_years`, trade the next `test_years`, roll forward, stitch OOS."""
    T = len(u.dates)
    train_n = int(train_years * TRADING_DAYS)
    test_n = int(test_years * TRADING_DAYS)
    folds = []
    oos_parts: List[pd.Series] = []
    gross_parts: List[pd.Series] = []
    start = min_warmup_days
    k = 0
    while start + train_n + 1 < T:
        tr0, tr1 = start, start + train_n
        te0, te1 = tr1, min(tr1 + test_n, T)
        k += 1
        LOG.info("STAGE C  WFO fold %d: train %s->%s | test %s->%s", k, u.dates[tr0].date(),
                 u.dates[tr1 - 1].date(), u.dates[te0].date(), u.dates[te1 - 1].date())
        gs = grid_search(u, ind, grid, tr0, tr1, decision_days, cost_bps, objective)
        res = run_engine(u, ind, gs.best, te0, te1, decision_days, cost_bps)
        rf = u.cash_ret[te0:te1]
        folds.append({"fold": k, "train_start": u.dates[tr0].date(), "train_end": u.dates[tr1 - 1].date(),
                      "test_start": u.dates[te0].date(), "test_end": u.dates[te1 - 1].date(),
                      **asdict(gs.best), "is_objective": gs.best_smoothed_score,
                      "oos_sharpe": fast_sharpe(res.returns.to_numpy(), rf),
                      "oos_return": float(np.prod(1 + res.returns.to_numpy()) - 1),
                      "bench_oos_return": float(np.prod(1 + u.ret[te0:te1, u.spy_idx]) - 1)})
        oos_parts.append(res.returns)
        gross_parts.append(res.gross)
        start += test_n
    if not oos_parts:
        raise RuntimeError("not enough data for a single walk-forward fold")
    return {"folds": pd.DataFrame(folds), "oos_returns": pd.concat(oos_parts),
            "oos_gross": pd.concat(gross_parts)}


# ----------------------------------------------------------------------------
# STAGE D - REPORTING HELPERS
# ----------------------------------------------------------------------------

def crisis_table(series: Dict[str, pd.Series], oos_start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for name, (a, b) in CRISIS_WINDOWS.items():
        a_ts, b_ts = pd.Timestamp(a), pd.Timestamp(b)
        row = {"window": name, "from": a, "to": b,
               "segment": "OOS" if a_ts >= oos_start else "IN-SAMPLE"}
        any_data = False
        for key, s in series.items():
            seg = s.loc[a_ts:b_ts]
            if len(seg) > 5:
                row[key] = float(np.prod(1 + seg.to_numpy()) - 1)
                any_data = True
            else:
                row[key] = np.nan
        if any_data:
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def yearly_table(series: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(series)
    yr = df.groupby(df.index.year).apply(lambda g: (1 + g).prod() - 1)
    return yr


def save_plot(series: Dict[str, pd.Series], path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        LOG.info("matplotlib not available; skipping chart")
        return False
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, s in series.items():
        eq = (1 + s).cumprod()
        axes[0].plot(eq.index, eq.values, label=name, lw=1.2)
        axes[1].plot(eq.index, (eq / eq.cummax() - 1).values, label=name, lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title(title)
    axes[0].set_ylabel("Growth of $1 (log)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def banner(text: str) -> None:
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--risk-assets", nargs="+", default=list(RISK_ASSETS))
    ap.add_argument("--cash-proxies", nargs="+", default=list(CASH_PROXIES))
    ap.add_argument("--rebalance", default="W", choices=["D", "W", "M"], help="decision frequency")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="one-way transaction cost (bps of turnover)")
    ap.add_argument("--vol-target", type=float, default=0.10)
    ap.add_argument("--bear-cap", type=float, default=0.75)
    ap.add_argument("--rebalance-band", type=float, default=0.05, help="no-trade band on held names (weight units)")
    ap.add_argument("--objective", default="sharpe", choices=["sharpe", "sortino", "calmar"])
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--satellite-weight", type=float, default=0.30, help="satellite share in the core+satellite blend")
    ap.add_argument("--wfo", action="store_true", help="also run rolling walk-forward optimisation")
    ap.add_argument("--wfo-train-years", type=float, default=6.0)
    ap.add_argument("--wfo-test-years", type=float, default=2.0)
    ap.add_argument("--fast", action="store_true", help="reduced grid for quick runs")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (offline smoke test)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--output-dir", default="satellite_output")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------------- STAGE A ----------------
    banner("STAGE A  DATA INGESTION & CLEANING")
    tickers = list(dict.fromkeys(list(args.risk_assets) + list(args.cash_proxies)))
    if args.synthetic:
        LOG.warning("SYNTHETIC DATA MODE - results are a smoke test, NOT a market backtest")
        prices = synthetic_prices(args.start, args.end)
    else:
        prices = fetch_prices(tickers, args.start, args.end, Path(args.cache_dir), use_cache=not args.no_cache)
    try:
        u = align_universe(prices, args.risk_assets, args.cash_proxies, BENCHMARK)
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 2
    print(f"Universe  : {u.tickers}   cash proxy: {u.cash_label}")
    print(f"History   : {u.dates[0].date()} -> {u.dates[-1].date()}   ({len(u.dates)} trading days)")

    # ---------------- STAGE B ----------------
    banner("STAGE B  SYSTEMATIC INDICATOR GENERATION")
    grid = default_grid(fast=args.fast)
    grid = [replace(p, vol_target=args.vol_target, bear_cap=args.bear_cap, rebalance_band=args.rebalance_band)
            for p in grid]
    ind = build_indicators(u, grid)
    decision_days = rebalance_mask(u.dates, args.rebalance)
    print(f"Indicators: SMA windows {sorted(ind.sma)} | momentum lookbacks {sorted(ind.mom)} | "
          f"vol window {sorted(ind.vol)} | ATR window {sorted(ind.atr)}")
    print(f"Decisions : {int(decision_days.sum())} decision days at frequency '{args.rebalance}', "
          f"executed next close, cost {args.cost_bps:g} bps one-way")
    print(f"Grid      : {len(grid)} parameter sets, objective = {args.objective}")

    # ---------------- STAGE C ----------------
    banner("STAGE C  ANTI-OVERFITTING ENGINE  (70/30 chronological holdout)")
    hv = holdout_validation(u, ind, grid, decision_days, args.cost_bps, args.objective, args.train_frac)
    split = hv["split"]
    best: StrategyParams = hv["best"]
    gs: GridResult = hv["grid"]
    oos_start = u.dates[split]
    print(f"Train     : {u.dates[0].date()} -> {u.dates[split - 1].date()}")
    print(f"Test (OOS): {oos_start.date()} -> {u.dates[-1].date()}")
    print(f"Selected  : {best.label()}  (vol_target={best.vol_target:g}, bear_cap={best.bear_cap:g})")
    print("\nTop 10 parameter sets on TRAIN (ranked by neighbourhood-smoothed objective):")
    show_cols = ["mom_lookback", "sma_window", "top_k", "stop_atr_mult", "crisis_cap", "objective",
                 "smoothed_objective", "sharpe", "cagr", "max_drawdown", "turnover_py", "n_stops", "avg_gross"]
    print(gs.table[show_cols].head(10).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    q = gs.table["sharpe"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print("\nParameter-surface diagnostics (TRAIN Sharpe across the whole grid):")
    print(f"  N trials = {len(gs.table)} | median = {q[0.5]:.2f} | IQR = [{q[0.25]:.2f}, {q[0.75]:.2f}] | "
          f"P10/P90 = {q[0.1]:.2f}/{q[0.9]:.2f} | share with Sharpe > 0.5 = {(gs.table['sharpe'] > 0.5).mean():.0%}")
    print(f"  Deflated Sharpe Ratio of the winner (in-sample, N={len(gs.table)} trials): "
          f"DSR = {hv['dsr']:.3f}   (expected max SR under the null, SR* = {hv['sr_star']:.2f})")
    print("  Read: DSR is P[true in-sample Sharpe > SR*]. Values below ~0.95 mean the in-sample edge is not\n"
          "        distinguishable from selection luck, so trust only the OOS block below.")
    gs.table.to_csv(out_dir / "grid_results_train.csv", index=False)

    # ---------------- STAGE D ----------------
    rf = pd.Series(u.cash_ret, index=u.dates, name="cash")
    risk_rets = pd.DataFrame(u.ret, index=u.dates, columns=u.tickers)
    spy = risk_rets[BENCHMARK].rename("SPY buy & hold")
    test_res: EngineResult = hv["test"]
    train_res: EngineResult = hv["train"]
    sat_oos = test_res.returns.rename("Satellite (OOS)")
    oos = slice(oos_start, None)
    blend_w = args.satellite_weight
    series_oos = {
        "SPY buy & hold": spy.loc[oos],
        "60/40 SPY/TLT (monthly)": rebalanced_mix(risk_rets.loc[oos, ["SPY", "TLT"]], [0.6, 0.4]).rename("60/40")
        if "TLT" in u.tickers else None,
        "Satellite (OOS)": sat_oos,
        f"Core {1 - blend_w:.0%} SPY + {blend_w:.0%} Satellite (monthly)":
            rebalanced_mix(pd.concat([spy.loc[oos], sat_oos], axis=1), [1 - blend_w, blend_w]),
    }
    series_oos = {k: v for k, v in series_oos.items() if v is not None}

    banner(f"STAGE D  OUT-OF-SAMPLE RESULTS  {oos_start.date()} -> {u.dates[-1].date()}  "
           f"(parameters frozen from TRAIN; never touched OOS data)")
    tab_oos = performance_table(series_oos, rf, "SPY buy & hold")
    print(format_table(tab_oos))
    print(f"\nSatellite OOS activity: {test_res.n_rebalances} decision days, {test_res.n_stops} stop-outs, "
          f"annualised one-way turnover {test_res.turnover / (len(sat_oos) / TRADING_DAYS):.2f}x, "
          f"avg gross risk {test_res.gross.mean():.0%}, time fully in cash {(test_res.gross <= 1e-9).mean():.0%}")
    print("Sharpe/Sortino are computed on returns in EXCESS of the cash proxy; recovery_days = trading days from\n"
          "the max-drawdown trough to the next equity high ('not yet' if still under water).")
    tab_oos.to_csv(out_dir / "summary_oos.csv")

    banner("STAGE D  IN-SAMPLE (TRAIN) RESULTS - for reference only; parameters were fitted here")
    series_is = {"SPY buy & hold": spy.iloc[:split], "Satellite (train)": train_res.returns.rename("Satellite (train)")}
    print(format_table(performance_table(series_is, rf, "SPY buy & hold")))

    banner("STAGE D  CRISIS WINDOWS  (full-sample run of the selected parameters; IN-SAMPLE rows were seen by the optimiser)")
    full_res = run_engine(u, ind, best, 0, len(u.dates), decision_days, args.cost_bps)
    crisis_series = {"SPY": spy, "Satellite": full_res.returns}
    if "TLT" in u.tickers:
        crisis_series["TLT"] = risk_rets["TLT"]
    if "GLD" in u.tickers:
        crisis_series["GLD"] = risk_rets["GLD"]
    ct = crisis_table(crisis_series, oos_start)
    if not ct.empty:
        fmt = ct.copy()
        for c in crisis_series:
            fmt[c] = fmt[c].map(lambda v: "n/a" if pd.isna(v) else f"{100 * v:+.1f}%")
        print(fmt.to_string(index=False))

    banner("STAGE D  CALENDAR-YEAR RETURNS (OOS segment)")
    print(yearly_table({k: v for k, v in series_oos.items()}).map(lambda v: f"{100 * v:+.1f}%").to_string())

    equity = pd.DataFrame({k: (1 + v).cumprod() for k, v in series_oos.items()})
    equity.to_csv(out_dir / "oos_equity_curves.csv")
    test_res.weights.to_csv(out_dir / "oos_weights.csv")
    chart = save_plot(series_oos, out_dir / "oos_equity.png", "Out-of-sample: satellite vs benchmarks")

    # ---------------- optional rolling WFO ----------------
    if args.wfo:
        banner("STAGE C/D  ROLLING WALK-FORWARD OPTIMISATION (stitched OOS folds)")
        wf = rolling_walk_forward(u, ind, grid, decision_days, args.cost_bps, args.objective,
                                  args.wfo_train_years, args.wfo_test_years)
        folds: pd.DataFrame = wf["folds"]
        fold_cols = ["fold", "test_start", "test_end", "mom_lookback", "sma_window", "top_k", "stop_atr_mult",
                     "crisis_cap", "is_objective", "oos_sharpe", "oos_return", "bench_oos_return"]
        print(folds[fold_cols].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        wfo_ret: pd.Series = wf["oos_returns"].rename("Satellite (WFO stitched)")
        series_wfo = {"SPY buy & hold": spy.reindex(wfo_ret.index), "Satellite (WFO stitched)": wfo_ret,
                      f"Core {1 - blend_w:.0%} SPY + {blend_w:.0%} Satellite (monthly)":
                          rebalanced_mix(pd.concat([spy.reindex(wfo_ret.index), wfo_ret], axis=1), [1 - blend_w, blend_w])}
        print(f"\nStitched OOS period {wfo_ret.index[0].date()} -> {wfo_ret.index[-1].date()} "
              f"({len(folds)} folds, parameters re-fitted every {args.wfo_test_years:g} years)")
        print(format_table(performance_table(series_wfo, rf, "SPY buy & hold")))
        stability = folds[["mom_lookback", "sma_window", "top_k", "stop_atr_mult", "crisis_cap"]].nunique()
        print("\nParameter stability across folds (distinct values chosen): " + ", ".join(f"{k}={v}" for k, v in stability.items()))
        folds.to_csv(out_dir / "wfo_folds.csv", index=False)
        pd.DataFrame({k: (1 + v).cumprod() for k, v in series_wfo.items()}).to_csv(out_dir / "wfo_equity_curves.csv")

    banner("DONE")
    files = sorted(p.name for p in out_dir.iterdir())
    print(f"Artifacts in {out_dir.resolve()}: {', '.join(files)}")
    if not chart:
        print("(install matplotlib to also get oos_equity.png)")
    print(f"Elapsed {time.time() - t_start:.0f}s")
    if args.synthetic:
        print("\n*** SYNTHETIC DATA WAS USED - do not interpret these numbers as market results ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
