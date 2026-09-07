"""Dealer gamma exposure (GEX) from a chain snapshot.

The mechanism this measures: dealers are (by the standard convention) long
the calls customers sold them and short the puts customers bought, so their
net gamma per strike is ``+call gamma x OI - put gamma x OI``. Where net GEX
is positive, dealer hedging leans against price (sell rallies, buy dips) and
suppresses moves; where negative, it chases price and amplifies them.

From that one table come the three numbers people quote:

- **net GEX** — total dollar gamma per 1% move; its sign is the regime.
- **call wall / put wall** — the strikes with the most call / put open
  interest, where hedging flow is concentrated.
- **gamma flip** — the level where per-strike net GEX changes sign, i.e.
  the boundary between the put-dominated (dealer short gamma) region below
  and the call-dominated (dealer long gamma) region above. When there are
  several crossings, the one nearest spot is reported.

Two honesty notes. The dealer-sign convention is an assumption, not a fact
observable from the chain; when retail is heavily long calls that have gone
in the money, dealers can be *short* those calls and the sign flips. And the
flip here is the strike-level approximation retail tools call "zero gamma" —
the exact flip requires re-pricing every contract's gamma at each candidate
spot, which needs a vol surface this module does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

from ingestion.schema import OptionRight, OptionsChainSnapshot

# One option contract controls this many shares.
_MULTIPLIER = 100

# GEX is conventionally quoted as dollar gamma per 1% move in the underlying.
_PCT_MOVE = 0.01


@dataclass(frozen=True)
class GexProfile:
    """Dealer gamma positioning for one underlying at one ``as_of``."""

    as_of: datetime
    underlying: str
    spot: float
    net_gex: float
    call_wall: float
    put_wall: float
    gamma_flip: float | None
    by_strike: pd.DataFrame
    n_used: int
    n_skipped: int

    @property
    def dealer_gamma(self) -> Literal["long", "short"]:
        """Regime: ``long`` suppresses moves (pinning), ``short`` amplifies them."""
        return "long" if self.net_gex > 0.0 else "short"


def gex_by_strike(chain: OptionsChainSnapshot) -> tuple[pd.DataFrame, int, int]:
    """Per-strike table of call/put GEX and open interest.

    Contracts with no gamma or zero open interest contribute nothing and are
    counted in the skipped total; the caller decides whether coverage is
    acceptable.

    Returns:
        ``(table, n_used, n_skipped)`` — the table is indexed by strike with
        columns ``call_gex, put_gex, net_gex, call_oi, put_oi``.
    """
    spot = chain.underlying_price
    scale = _MULTIPLIER * spot * spot * _PCT_MOVE
    rows = []
    n_skipped = 0
    for c in chain.contracts:
        if c.gamma is None or c.open_interest <= 0:
            n_skipped += 1
            continue
        dollar_gamma = c.gamma * c.open_interest * scale
        is_call = c.right is OptionRight.call
        rows.append(
            {
                "strike": c.strike,
                "call_gex": dollar_gamma if is_call else 0.0,
                "put_gex": -dollar_gamma if not is_call else 0.0,
                "call_oi": c.open_interest if is_call else 0,
                "put_oi": c.open_interest if not is_call else 0,
            }
        )
    if not rows:
        empty = pd.DataFrame(columns=["call_gex", "put_gex", "net_gex", "call_oi", "put_oi"])
        return empty, 0, n_skipped
    table = pd.DataFrame(rows).groupby("strike").sum().sort_index()
    table["net_gex"] = table["call_gex"] + table["put_gex"]
    return table[["call_gex", "put_gex", "net_gex", "call_oi", "put_oi"]], len(rows), n_skipped


def _gamma_flip(table: pd.DataFrame, spot: float) -> float | None:
    """Zero crossing of per-strike net GEX nearest to spot; None if one-signed."""
    strikes = table.index.to_numpy(dtype=float)
    values = table["net_gex"].to_numpy(dtype=float)
    crossings: list[float] = []
    for i in range(1, len(values)):
        lo, hi = values[i - 1], values[i]
        if lo == 0.0 and hi == 0.0:
            continue
        if lo == 0.0 or (lo < 0.0) != (hi < 0.0):
            # Linear interpolation between the two strikes bracketing the crossing.
            frac = 0.0 if lo == 0.0 else lo / (lo - hi)
            crossings.append(float(strikes[i - 1] + frac * (strikes[i] - strikes[i - 1])))
    if not crossings:
        return None
    return min(crossings, key=lambda x: abs(x - spot))


def gex_profile(chain: OptionsChainSnapshot) -> GexProfile:
    """Net GEX, walls and flip for one chain snapshot.

    Raises:
        ValueError: If no contract carries both a gamma and open interest —
            there is nothing to measure, and a profile of zeros would be a
            silent lie.
    """
    table, n_used, n_skipped = gex_by_strike(chain)
    if n_used == 0:
        msg = (
            f"{chain.underlying} @ {chain.as_of.isoformat()}: no contract has both "
            f"gamma and open interest ({n_skipped} skipped)"
        )
        raise ValueError(msg)
    return GexProfile(
        as_of=chain.as_of,
        underlying=chain.underlying,
        spot=chain.underlying_price,
        net_gex=float(table["net_gex"].sum()),
        call_wall=float(table["call_oi"].idxmax()),
        put_wall=float(table["put_oi"].idxmax()),
        gamma_flip=_gamma_flip(table, chain.underlying_price),
        by_strike=table,
        n_used=n_used,
        n_skipped=n_skipped,
    )
