"""Expiry-pin hypothesis: does price get dragged toward a gamma wall into expiry?

Run from ``quant-stack/``:

    uv run python examples/pin_hypothesis.py              # live 1h bars via Yahoo
    uv run python examples/pin_hypothesis.py --synthetic  # offline plumbing check

The hypothesis (see the conversation / README for the full statement):

    On option-expiry days, dealers who are net LONG gamma at a heavily-traded
    strike hedge by selling above it and buying below it. That flow drags
    the close toward the strike. The counterparty is the directional option
    buyer who owns near-the-money contracts into expiry; they lose because
    the move they paid for is suppressed by the hedging their own
    positioning created, and their premium decays to the dealers.

This script tests the NECESSARY condition on spot: from ``start_hour_et`` on
an expiry day, take a position that fades the distance to the nearest pin
strike (long below it, short above it, scaled by distance), flat overnight.
If spot is not pulled toward the strike, no options structure on it works.
It does NOT model options P&L (theta, vega, assignment) — that comes after,
and only if this passes.

Simplifications to be honest about:

- ``PINS`` is a fixed candidate list applied to every Friday. Real
  per-expiry open interest should replace it. Picking pins after seeing the
  close is hindsight and invalidates the test.
- Expiry days are Fridays (weeklies). Holiday-shifted expiries are ignored.
- Bars are hourly (Yahoo's ~730-day limit); the engine acts on the NEXT bar,
  so a 12:00 signal is a 13:00 position. Finer bars tighten this.
"""

from __future__ import annotations

import argparse
import itertools
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

SYMBOL = "IREN"
PINS = (45.0, 50.0)  # candidate gamma walls — replace with real per-expiry OI
BANDS = (0.02, 0.03, 0.05)  # fade within +/- band of the pin; escape at 2x band
START_HOURS_ET = (11, 12, 13)  # begin fading from this hour on expiry day
EXPIRY_WEEKDAY = 4  # Friday
BARS_PER_DAY = 7
PPY = 252 * BARS_PER_DAY  # hourly bars per year
CFG = BacktestConfig(fee_bps=2.0, slippage_bps=5.0, periods_per_year=PPY)
WF = WalkForwardConfig(train_bars=120 * BARS_PER_DAY, test_bars=30 * BARS_PER_DAY)
LIVE_DEMO_BARS = 10 * BARS_PER_DAY


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_prices(synthetic: bool) -> pd.Series:
    if synthetic:
        print("SYNTHETIC DATA — a random walk around 47, not IREN. Numbers mean nothing.")
        days = pd.bdate_range("2024-09-01", periods=500)
        stamps = [
            pd.Timestamp(f"{d.date()} {h:02d}:{m:02d}", tz="America/New_York")
            for d in days
            for (h, m) in ((10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30), (16, 0))
        ]
        idx = pd.DatetimeIndex(stamps).tz_convert("UTC")
        rng = np.random.default_rng(7)
        return pd.Series(47.0 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx)))), index=idx)
    with YahooClient() as yc:
        bars = yc.get_history(SYMBOL, period="2y", interval="1h")
    idx = pd.DatetimeIndex([b.as_of for b in bars])
    return pd.Series([b.close for b in bars], index=idx, name=SYMBOL)


def pin_signal(band: float, start_hour_et: int) -> FittedStrategy:
    """Fade the distance to the nearest pin on expiry afternoons; flat otherwise."""

    def signal(history: pd.Series) -> pd.Series:
        et = history.index.tz_convert("America/New_York")
        in_window = (et.dayofweek == EXPIRY_WEEKDAY) & (et.hour >= start_hour_et)

        p = history.to_numpy(dtype=float)
        pins = np.asarray(PINS, dtype=float)
        rel = p[:, None] / pins[None, :] - 1.0  # distance to each pin, as a fraction
        nearest = np.abs(rel).argmin(axis=1)
        d = rel[np.arange(len(p)), nearest]  # + above the pin, - below it

        escaped = np.abs(d) > 2.0 * band  # hedging lost to flow: stand aside
        # Long below the pin, short above it, scaled by distance within the band.
        fade = np.clip(-d / band, -1.0, 1.0)
        sig = np.where(in_window & ~escaped, fade, 0.0)
        return pd.Series(sig, index=history.index)

    return FittedStrategy(
        signal_fn=signal, params={"band": band, "start_hour_et": float(start_hour_et)}
    )


VARIANTS = tuple(itertools.product(BANDS, START_HOURS_ET))


def fit_variant(train: pd.Series) -> FittedStrategy:
    """Pick (band, start hour) by IN-SAMPLE Sharpe on the train window only."""
    best, best_sharpe = VARIANTS[0], -math.inf
    for band, hour in VARIANTS:
        sharpe = run_backtest(train, pin_signal(band, hour).signal_fn(train), CFG).metrics.sharpe
        if not math.isnan(sharpe) and sharpe > best_sharpe:
            best, best_sharpe = (band, hour), sharpe
    return pin_signal(*best)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--synthetic", action="store_true", help="skip the network")
    args = parser.parse_args()

    banner("1. Data")
    prices = load_prices(args.synthetic)
    n_expiry = int((prices.index.tz_convert("America/New_York").dayofweek == EXPIRY_WEEKDAY).sum())
    print(f"{len(prices)} hourly bars, {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"last close: {prices.iloc[-1]:,.2f}   pins: {PINS}   expiry-day bars: {n_expiry}")

    banner(f"2. Try N={len(VARIANTS)} variations (band x start hour), then deflate the best")
    trials: dict[tuple[float, int], float] = {}
    results = {}
    for band, hour in VARIANTS:
        res = run_backtest(prices, pin_signal(band, hour).signal_fn(prices), CFG)
        results[(band, hour)] = res
        trials[(band, hour)] = res.metrics.sharpe
        m = res.metrics
        print(f"  band {band:.0%} from {hour:02d}:00  sharpe {m.sharpe:6.2f}  "
              f"return {m.total_return:7.1%}  trades {m.num_trades:4d}  hit {m.hit_rate:5.1%}")
    best = max(trials, key=lambda k: trials[k] if not math.isnan(trials[k]) else -math.inf)
    trial_pp = [deannualize_sharpe(s, PPY) for s in trials.values() if not math.isnan(s)]
    if math.isnan(trials[best]) or not trial_pp:
        print("\n  every variant was flat (no expiry-day bars within a band) — nothing to deflate")
    else:
        dsr = deflated_sharpe(results[best].ledger["net"], trial_sharpes=trial_pp)
        print(f"\n  best: band {best[0]:.0%} from {best[1]:02d}:00 (annualized {trials[best]:.2f})")
        print(f"  deflated Sharpe = {dsr.deflated_sharpe:.3f}  ->  "
              f"{'PASSES' if dsr.passes else 'NOT distinguishable from noise'} at 0.95")

    banner("3. Walk-forward (variant chosen on each train window only)")
    wf = walk_forward(prices, fit_variant, CFG, WF)
    s = wf.summary()
    print(f"  {wf.n_folds} folds, positive {s['positive_folds']}/{s['n_folds']}")
    print(f"  stitched OOS sharpe {s['oos_sharpe']:.2f}  return {s['oos_total_return']:.1%}"
          f"  maxdd {s['oos_max_drawdown']:.1%}")
    print("  band chosen per fold:      ", wf.param_table["band"].tolist())
    print("  start hour chosen per fold:", wf.param_table["start_hour_et"].astype(int).tolist())

    banner("4. Where did the OOS P&L come from?")
    rep = regime_report(wf.result, prices, RegimeConfig(ma_bars=200, chop_band=0.02))
    print(rep.table[["n_bars", "time_share", "sharpe", "total_return", "pnl_share"]]
          .to_string(float_format=lambda x: f"{x:7.2f}"))
    print(f"\n  {rep.verdict()}")

    banner("5. Size the spot-proxy trade off the latest close")
    entry = float(prices.iloc[-1])
    band = float(wf.param_table["band"].iloc[-1])
    stop = entry * (1.0 - 2.0 * band)  # escape level = the trade's invalidation
    ps = position_size(10_000, entry, stop, Side.long, SizingConfig(risk_fraction=0.005))
    print(f"  long {ps.units:.2f} sh @ {entry:.2f}, stop {stop:.2f} (2x band escape)")
    print(f"  notional ${ps.notional:,.0f} ({ps.position_fraction:.1%})   "
          f"loss if stopped ${ps.loss_if_stopped:,.0f} = {ps.risk_fraction_realized:.2%}")

    banner(f"6. Health check — pretending the last {LIVE_DEMO_BARS} OOS bars are live")
    oos = wf.result
    bench = Benchmark(
        sharpe_per_period=deannualize_sharpe(oos.metrics.sharpe, PPY),
        max_drawdown=oos.metrics.max_drawdown,
        longest_underwater_bars=longest_underwater(oos.ledger["equity"]),
    )
    live = oos.ledger["net"].iloc[-LIVE_DEMO_BARS:]
    hc = health_check(live, bench, HealthConfig(window=35, periods_per_year=PPY))
    print(f"  live sharpe {hc.live_sharpe_annualized:.2f} +/- {hc.live_sharpe_se:.2f} (SE)   "
          f"p(consistent) {hc.p_consistent:.2f}")
    print(f"  alerts: {list(hc.alerts) or 'none'}   ->   recommendation: {hc.recommendation}")
    print("\n  (It recommends. You press the button.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
