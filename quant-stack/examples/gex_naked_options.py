"""GEX-informed, naked-only options strategy — the post's idea, made testable.

Run from ``quant-stack/``:

    uv run python examples/gex_naked_options.py --synthetic

There is no live mode yet: this needs chain snapshots with open interest and
gamma, which come from your Schwab capture (``ParquetStore.write_options_chain``)
or an OptionsDX backfill. Until those exist, ``--synthetic`` runs the whole
chain on Black-Scholes chains over a random walk so you can see the plumbing.
Every number it prints on synthetic data is a plumbing check, not evidence.

The hypothesis, in naked-only terms:

    Dealers net LONG gamma pin price between the walls; dealers net SHORT
    gamma amplify moves through the flip. The counterparty is the directional
    option buyer whose premium decays while price is pinned.

    - Long-gamma regime, spot within ``band`` above the put wall: sell ONE
      cash-secured put at the put wall (collect the premium the pin decays).
      Invalidation: spot closes below the put wall -> buy back. Otherwise it
      settles at expiry.
    - Short-gamma regime, spot below the gamma flip: buy ONE ATM put (moves
      are amplified below the flip). Exit: spot back above the flip.

Only cash-secured puts and long puts — both inside the schema's allowlist.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from backtest import (
    Benchmark,
    HealthConfig,
    RegimeConfig,
    deannualize_sharpe,
    deflated_sharpe,
    health_check,
    longest_underwater,
    regime_report,
)
from ingestion.schema import OptionRight, OptionsChainSnapshot
from options import (
    Book,
    ContractKey,
    Portfolio,
    SimConfig,
    StrategyFn,
    gex_profile,
    simulate,
    synthetic_chain,
)

UNDERLYING = "QQQ"
BANDS = (0.01, 0.02, 0.03)  # how close to the put wall counts as "at the wall"
MIN_DTE = (5, 10)  # shortest expiry the strategy will sell
VARIANTS = tuple(itertools.product(BANDS, MIN_DTE))
SIM = SimConfig(initial_capital=100_000.0, commission_per_contract=0.65, periods_per_year=252)
LIVE_DEMO_BARS = 40


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _next_fridays(from_day: date, n: int) -> list[date]:
    """The next ``n`` Friday expiries, INCLUDING today if today is a Friday —
    real chains list expiring contracts until the close."""
    d = from_day + timedelta(days=(4 - from_day.weekday()) % 7)
    return [d + timedelta(weeks=i) for i in range(n)]


def _synthetic_oi(spot: float) -> Callable[[float, OptionRight], int]:
    """OI concentrated on round strikes near the money, calls a little heavier.

    Puts pile up below spot and calls above — the usual shape — decaying
    with distance so the walls sit near the money. Calls are weighted up so
    dealers are net long gamma most of the time, the common index state.
    """

    def oi(k: float, r: OptionRight) -> int:
        proximity = math.exp(-abs(k - spot) / (0.05 * spot))
        base = (3000 if k % 25 == 0 else 300) * proximity
        if r is OptionRight.put:
            return int(base if k <= spot else base / 3)
        return int(1.3 * (base if k >= spot else base / 3))

    return oi


def synthetic_history(n_days: int, seed: int = 11) -> list[OptionsChainSnapshot]:
    """Daily Black-Scholes chains over a random walk."""
    print("SYNTHETIC CHAINS — Black-Scholes over a random walk. Plumbing check only.")
    rng = np.random.default_rng(seed)
    spot = 500.0
    day = date(2026, 1, 5)
    out: list[OptionsChainSnapshot] = []
    for _ in range(n_days):
        if day.weekday() < 5:
            spot *= math.exp(rng.normal(0.0003, 0.012))
            lo, hi = int(spot * 0.9) // 5 * 5, int(spot * 1.1) + 5
            strikes = [float(k) for k in range(lo, hi, 5)]
            as_of = datetime(day.year, day.month, day.day, 21, 0, tzinfo=UTC)
            out.append(
                synthetic_chain(
                    as_of, UNDERLYING, round(spot, 2), _next_fridays(day, 4), strikes,
                    iv=0.22, open_interest=_synthetic_oi(spot),
                )
            )
        day += timedelta(days=1)
    return out


def make_strategy(band: float, min_dte: int) -> StrategyFn:
    def strategy(chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
        book = pf.book
        prof = gex_profile(chain)
        spot = prof.spot
        today = chain.as_of.date()
        expiries = sorted(
            {c.expiration for c in chain.contracts if (c.expiration - today).days >= min_dte}
        )
        if not expiries:
            return book
        exp = expiries[0]

        # Manage what's open first.
        for key, qty in list(book.options.items()):
            if qty < 0 and spot < key.strike:
                book = book.with_option(key, 0)  # CSP invalidated: buy back
            elif qty > 0 and prof.gamma_flip is not None and spot > prof.gamma_flip:
                book = book.with_option(key, 0)  # long-put thesis over
        if book.options:
            return book

        if prof.dealer_gamma == "long":
            wall = prof.put_wall
            # Only if the account can actually secure it — the simulator would
            # refuse otherwise, and refusing is the right thing for it to do.
            if 0.0 <= spot / wall - 1.0 <= band and pf.can_secure_puts(wall):
                return book.with_option(ContractKey(exp, wall, OptionRight.put), -1)
        elif prof.gamma_flip is not None and spot < prof.gamma_flip:
            strikes = sorted({c.strike for c in chain.contracts if c.expiration == exp})
            atm = min(strikes, key=lambda k: abs(k - spot))
            return book.with_option(ContractKey(exp, atm, OptionRight.put), 1)
        return book

    return strategy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--synthetic", action="store_true", help="run on Black-Scholes chains")
    args = parser.parse_args()
    if not args.synthetic:
        print("No live chain source is wired yet (needs Schwab capture or OptionsDX). "
              "Run with --synthetic to exercise the plumbing.")
        return 2

    banner("1. Data")
    chains = synthetic_history(400)
    spots = pd.Series(
        [c.underlying_price for c in chains], index=pd.DatetimeIndex([c.as_of for c in chains])
    )
    prof = gex_profile(chains[-1])
    flip = "none" if prof.gamma_flip is None else f"{prof.gamma_flip:.1f}"
    print(f"{len(chains)} daily chains, {len(chains[-1].contracts)} contracts in the last one")
    print(f"last: spot {prof.spot:.2f}  dealer gamma {prof.dealer_gamma}  "
          f"call wall {prof.call_wall:.0f}  put wall {prof.put_wall:.0f}  flip {flip}")

    banner(f"2. Try N={len(VARIANTS)} variations (band x min DTE), then deflate the best")
    runs = {}
    for band, dte in VARIANTS:
        sim = simulate(chains, make_strategy(band, dte), SIM)
        runs[(band, dte)] = sim
        m = sim.result.metrics
        print(f"  band {band:.0%} dte>={dte:<3} sharpe {m.sharpe:6.2f}  "
              f"return {m.total_return:7.1%}  maxdd {m.max_drawdown:7.1%}  "
              f"fills {len(sim.fills):3d}  settled {len(sim.settlements):3d}")
    usable = {k: v.result.metrics.sharpe for k, v in runs.items()
              if not math.isnan(v.result.metrics.sharpe)}
    if usable:
        best = max(usable, key=lambda k: usable[k])
        trial_pp = [deannualize_sharpe(s, SIM.periods_per_year) for s in usable.values()]
        dsr = deflated_sharpe(runs[best].result.ledger["net"], trial_sharpes=trial_pp)
        print(f"\n  best: band {best[0]:.0%} dte>={best[1]} (annualized {usable[best]:.2f})")
        print(f"  deflated Sharpe = {dsr.deflated_sharpe:.3f}  ->  "
              f"{'PASSES' if dsr.passes else 'NOT distinguishable from noise'} at 0.95")
    else:
        best = VARIANTS[0]
        print("\n  every variant was flat — nothing to deflate")

    banner("3. Where did the P&L come from?")
    rep = regime_report(runs[best].result, spots, RegimeConfig(ma_bars=50, chop_band=0.01))
    print(rep.table[["n_bars", "time_share", "sharpe", "total_return", "pnl_share"]]
          .to_string(float_format=lambda x: f"{x:7.2f}"))
    print(f"\n  {rep.verdict()}")

    banner(f"4. Health check — pretending the last {LIVE_DEMO_BARS} bars are live")
    res = runs[best].result
    bench = Benchmark(
        sharpe_per_period=deannualize_sharpe(res.metrics.sharpe, SIM.periods_per_year),
        max_drawdown=res.metrics.max_drawdown,
        longest_underwater_bars=longest_underwater(res.ledger["equity"]),
    )
    hc = health_check(res.ledger["net"].iloc[-LIVE_DEMO_BARS:], bench, HealthConfig(window=30))
    print(f"  live sharpe {hc.live_sharpe_annualized:.2f} +/- {hc.live_sharpe_se:.2f} (SE)   "
          f"p(consistent) {hc.p_consistent:.2f}")
    print(f"  alerts: {list(hc.alerts) or 'none'}   ->   recommendation: {hc.recommendation}")
    print("\n  (It recommends. You press the button.)")
    print("\nNot yet covered: walk-forward for options (needs a chain-aware driver),")
    print("early assignment, and real skew/bid-ask — all of which need real chains first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
