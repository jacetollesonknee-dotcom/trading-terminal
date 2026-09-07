"""Black-Scholes price and greeks for a European option.

Used to mark *synthetic* chains and as a sanity reference. It is not a
substitute for a real quote: it has no skew, no bid/ask, no early exercise
(American equity options), and assumes constant vol. The simulator marks
real chains at their quoted mid, never at a model price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from ingestion.schema import OptionRight


@dataclass(frozen=True)
class Greeks:
    """Price and first-order sensitivities of one contract (per share)."""

    price: float
    delta: float
    gamma: float
    theta: float  # per year
    vega: float  # per 1.0 (100 vol points)


def _validate(spot: float, strike: float, t_years: float, vol: float) -> None:
    if spot <= 0.0:
        msg = f"spot must be > 0, got {spot}"
        raise ValueError(msg)
    if strike <= 0.0:
        msg = f"strike must be > 0, got {strike}"
        raise ValueError(msg)
    if t_years < 0.0:
        msg = f"t_years must be >= 0, got {t_years}"
        raise ValueError(msg)
    if vol <= 0.0:
        msg = f"vol must be > 0, got {vol}"
        raise ValueError(msg)


def bs_greeks(
    spot: float,
    strike: float,
    t_years: float,
    vol: float,
    right: OptionRight,
    rate: float = 0.0,
) -> Greeks:
    """Black-Scholes price and greeks. At ``t_years == 0`` returns intrinsic value."""
    _validate(spot, strike, t_years, vol)
    is_call = right is OptionRight.call
    if t_years == 0.0:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        in_the_money = (spot > strike) if is_call else (spot < strike)
        delta = (1.0 if is_call else -1.0) if in_the_money else 0.0
        return Greeks(price=intrinsic, delta=delta, gamma=0.0, theta=0.0, vega=0.0)

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = math.exp(-rate * t_years)
    pdf_d1 = float(norm.pdf(d1))
    cdf_d1 = float(norm.cdf(d1))
    cdf_d2 = float(norm.cdf(d2))

    if is_call:
        price = spot * cdf_d1 - strike * disc * cdf_d2
        delta = cdf_d1
        theta = -(spot * pdf_d1 * vol) / (2.0 * sqrt_t) - rate * strike * disc * cdf_d2
    else:
        price = strike * disc * (1.0 - cdf_d2) - spot * (1.0 - cdf_d1)
        delta = cdf_d1 - 1.0
        theta = -(spot * pdf_d1 * vol) / (2.0 * sqrt_t) + rate * strike * disc * (1.0 - cdf_d2)

    gamma = pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t
    return Greeks(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega)


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    vol: float,
    right: OptionRight,
    rate: float = 0.0,
) -> float:
    """Black-Scholes price only."""
    return bs_greeks(spot, strike, t_years, vol, right, rate).price
