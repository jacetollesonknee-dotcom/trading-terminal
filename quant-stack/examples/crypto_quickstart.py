"""Quickstart: the whole crypto backtest chain on real BTC-USD daily bars.

Run from ``quant-stack/``:

    uv run python examples/crypto_quickstart.py            # live Yahoo data
    uv run python examples/crypto_quickstart.py --synthetic # no network needed

What it does, in order — each step is one call into ``backtest/``:

1. Pull ~5 years of BTC-USD daily closes (Yahoo, via the repo's client).
2. Try N variations of a moving-average strategy, deflate the best one's
   Sharpe for having tried N.
3. Walk-forward: pick the lookback on each train window only, trade the
   unseen test window, stitch the out-of-sample result.
4. Attribute the OOS P&L to bull / bear / chop regimes.
5. Size a trade off the latest close with an honest stop.
6. Health-check the most recent bars as if they were live.

The strategy is deliberately dumb (long when close > moving average). The
point is the plumbing, not the edge. Expect the deflated Sharpe to say so.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from backtest import (
    BacktestConfig,
    Benchmark,
    FittedStrategy,
    HealthConfig,
    RegimeConfig,
    Side,
    SizingConfig,
    WalkForwardConfig,
    deannualize_sharpe,
    deflated_sharpe,
    health_check,
    longest_underwater,
    position_size,
    regime_report,
    run_backtest,
    walk_forward,
)
from ingestion.yahoo_client import YahooClient

SYMBOL = "BTC-USD"
CANDIDATE_LOOKBACKS = (20, 50, 100)
CFG = BacktestConfig(fee_bps=5.0, slippage_bps=3.0, periods_per_year=365)
WF = WalkForwardConfig(train_bars=365, test_bars=90)
LIVE_DEMO_BARS = 60


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_prices(synthetic: bool) -> pd.Series:
    if synthetic:
        print("SYNTHETIC DATA — a random walk, not BTC. Numbers below mean nothing.")
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=1825, freq="D", tz="UTC")
        return pd.Series(30_000 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, 1825))), index=idx)
    with YahooClient() as yc:
        bars = yc.get_history(SYMBOL, period="5y", interval="1d")
    idx = pd.DatetimeIndex([b.as_of for b in bars])
    return pd.Series([b.close for b in bars], index=idx, name=SYMBOL)


def ma_signal(lookback: int) -> FittedStrategy:
    """Long when close is above its trailing MA, flat otherwise; NaN in warm-up."""

    def signal(history: pd.Series) -> pd.Series:
        ma = history.rolling(lookback).mean()
        return (history > ma).astype(float).where(ma.notna())

    return FittedStrategy(signal_fn=signal, params={"lookback": float(lookback)})


def fit_lookback(train: pd.Series) -> FittedStrategy:
    """Pick the lookback with the best IN-SAMPLE Sharpe on the train window only."""
    best_lb, best_sharpe = CANDIDATE_LOOKBACKS[0], -math.inf
    for lb in CANDIDATE_LOOKBACKS:
        fitted = ma_signal(lb)
        sharpe = run_backtest(train, fitted.signal_fn(train), CFG).metrics.sharpe
        if not math.isnan(sharpe) and sharpe > best_sharpe:
            best_lb, best_sharpe = lb, sharpe
    return ma_signal(best_lb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--synthetic", action="store_true", help="skip the network")
    args = parser.parse_args()

    banner("1. Data")
    prices = load_prices(args.synthetic)
    print(f"{len(prices)} daily bars, {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"last close: {prices.iloc[-1]:,.2f}")

    banner(f"2. Try N={len(CANDIDATE_LOOKBACKS)} variations, then deflate the best")
    trials: dict[int, float] = {}
    results = {}
    for lb in CANDIDATE_LOOKBACKS:
        fitted = ma_signal(lb)
        res = run_backtest(prices, fitted.signal_fn(prices), CFG)
        results[lb] = res
        trials[lb] = res.metrics.sharpe
        m = res.metrics
        print(f"  MA{lb:<4} sharpe {m.sharpe:6.2f}  cagr {m.cagr:7.1%}  maxdd {m.max_drawdown:7.1%}"
              f"  trades {m.num_trades:4d}")
    best_lb = max(trials, key=lambda k: trials[k] if not math.isnan(trials[k]) else -math.inf)
    trial_pp = [deannualize_sharpe(s, CFG.periods_per_year) for s in trials.values()]
    dsr = deflated_sharpe(results[best_lb].ledger["net"], trial_sharpes=trial_pp)
    print(f"\n  best: MA{best_lb} (annualized Sharpe {trials[best_lb]:.2f})")
    print(f"  deflated Sharpe = {dsr.deflated_sharpe:.3f}  ->  "
          f"{'PASSES' if dsr.passes else 'NOT distinguishable from noise'} at 0.95")

    banner("3. Walk-forward (lookback chosen on each train window only)")
    wf = walk_forward(prices, fit_lookback, CFG, WF)
    s = wf.summary()
    print(f"  {wf.n_folds} folds, positive {s['positive_folds']}/{s['n_folds']}")
    print(f"  stitched OOS sharpe {s['oos_sharpe']:.2f}  cagr {s['oos_cagr']:.1%}"
          f"  maxdd {s['oos_max_drawdown']:.1%}")
    print("  lookback chosen per fold:", wf.param_table["lookback"].astype(int).tolist())

    banner("4. Where did the OOS P&L come from?")
    rep = regime_report(wf.result, prices, RegimeConfig(ma_bars=200, chop_band=0.02))
    print(rep.table[["n_bars", "time_share", "sharpe", "total_return", "pnl_share"]]
          .to_string(float_format=lambda x: f"{x:7.2f}"))
    print(f"\n  {rep.verdict()}")

    banner("5. Size a trade off the latest close (5% stop, 1% risk)")
    entry = float(prices.iloc[-1])
    ps = position_size(10_000, entry, entry * 0.95, Side.long,
                       SizingConfig(risk_fraction=0.01, lot_step=0.0001))
    print(f"  {ps.units:.4f} units  notional ${ps.notional:,.0f}  "
          f"({ps.position_fraction:.1%} of capital)")
    print(f"  loss if stopped ${ps.loss_if_stopped:,.0f} = {ps.risk_fraction_realized:.2%}"
          f"   capped={ps.capped} floored={ps.floored}")

    banner(f"6. Health check — pretending the last {LIVE_DEMO_BARS} OOS bars are live")
    oos = wf.result
    bench = Benchmark(
        sharpe_per_period=deannualize_sharpe(oos.metrics.sharpe, CFG.periods_per_year),
        max_drawdown=oos.metrics.max_drawdown,
        longest_underwater_bars=longest_underwater(oos.ledger["equity"]),
    )
    live = oos.ledger["net"].iloc[-LIVE_DEMO_BARS:]
    hc = health_check(live, bench, HealthConfig(window=30, consecutive_required=3))
    print(f"  live sharpe {hc.live_sharpe_annualized:.2f} +/- {hc.live_sharpe_se:.2f} (SE)")
    print(f"  p(consistent with benchmark) {hc.p_consistent:.2f}   "
          f"drawdown {hc.current_drawdown:.1%}   underwater {hc.bars_underwater} bars")
    print(f"  alerts: {list(hc.alerts) or 'none'}   ->   recommendation: {hc.recommendation}")
    print("\n  (It recommends. You press the button.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
