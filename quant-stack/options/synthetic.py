"""Black-Scholes chain generator — for tests and offline demos ONLY.

Everything it produces is model-priced: flat vol, no skew, a mechanical
spread, and open interest from whatever function you hand it. It exists so
the simulator and GEX code can be exercised without real chains. Any result
computed on a synthetic chain is a plumbing check, not evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime

from ingestion.schema import OptionContract, OptionRight, OptionsChainSnapshot
from options.pricing import bs_greeks

# Floor on the quoted spread so deep OTM contracts still show a market.
_MIN_SPREAD = 0.02
_SPREAD_FRACTION = 0.02
_DAYS_PER_YEAR = 365.0


def synthetic_chain(
    as_of: datetime,
    underlying: str,
    spot: float,
    expirations: Sequence[date],
    strikes: Sequence[float],
    iv: float = 0.25,
    open_interest: Callable[[float, OptionRight], int] | None = None,
) -> OptionsChainSnapshot:
    """Build a model-priced chain snapshot.

    Args:
        as_of: Snapshot time (UTC, tz-aware — the schema enforces it).
        underlying: Ticker.
        spot: Underlying price.
        expirations: Expiry dates (must be on/after ``as_of.date()``).
        strikes: Strike grid.
        iv: Flat implied vol applied to every contract.
        open_interest: ``(strike, right) -> OI``. Defaults to 100 everywhere.
    """
    oi_fn = open_interest or (lambda _k, _r: 100)
    contracts: list[OptionContract] = []
    for exp in expirations:
        days = (exp - as_of.date()).days
        if days < 0:
            msg = f"expiration {exp} is before as_of {as_of.date()}"
            raise ValueError(msg)
        t = days / _DAYS_PER_YEAR
        for k in strikes:
            for right in (OptionRight.call, OptionRight.put):
                g = bs_greeks(spot, k, t, iv, right)
                half = max(_MIN_SPREAD, g.price * _SPREAD_FRACTION) / 2.0
                bid = max(0.0, g.price - half)
                ask = g.price + half
                contracts.append(
                    OptionContract(
                        as_of=as_of,
                        underlying=underlying,
                        expiration=exp,
                        strike=k,
                        right=right,
                        bid=round(bid, 4),
                        ask=round(ask, 4),
                        last=round(g.price, 4),
                        open_interest=oi_fn(k, right),
                        iv=iv,
                        delta=g.delta,
                        gamma=g.gamma,
                        theta=g.theta,
                        vega=g.vega,
                        source="yahoo",
                    )
                )
    return OptionsChainSnapshot(
        as_of=as_of,
        underlying=underlying,
        underlying_price=spot,
        contracts=tuple(contracts),
        source="yahoo",
    )
