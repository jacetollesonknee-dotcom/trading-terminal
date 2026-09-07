"""Black-Scholes sanity tests."""

from __future__ import annotations

import math

import pytest

from ingestion.schema import OptionRight
from options import bs_greeks, bs_price


def test_put_call_parity() -> None:
    s, k, t, v, r = 100.0, 95.0, 0.5, 0.3, 0.02
    c = bs_price(s, k, t, v, OptionRight.call, r)
    p = bs_price(s, k, t, v, OptionRight.put, r)
    assert c - p == pytest.approx(s - k * math.exp(-r * t), abs=1e-9)


def test_call_above_intrinsic_and_below_spot() -> None:
    c = bs_price(100.0, 90.0, 0.25, 0.2, OptionRight.call)
    assert 10.0 < c < 100.0


def test_greeks_signs() -> None:
    g = bs_greeks(100.0, 100.0, 0.25, 0.2, OptionRight.call)
    assert 0.0 < g.delta < 1.0
    assert g.gamma > 0.0
    assert g.vega > 0.0
    assert g.theta < 0.0
    p = bs_greeks(100.0, 100.0, 0.25, 0.2, OptionRight.put)
    assert -1.0 < p.delta < 0.0
    assert p.gamma == pytest.approx(g.gamma)


def test_expiry_is_intrinsic() -> None:
    assert bs_price(110.0, 100.0, 0.0, 0.2, OptionRight.call) == 10.0
    assert bs_price(90.0, 100.0, 0.0, 0.2, OptionRight.call) == 0.0
    assert bs_price(90.0, 100.0, 0.0, 0.2, OptionRight.put) == 10.0
    g = bs_greeks(110.0, 100.0, 0.0, 0.2, OptionRight.call)
    assert g.delta == 1.0 and g.gamma == 0.0


@pytest.mark.parametrize(
    ("spot", "strike", "t", "vol"),
    [
        (0.0, 100.0, 1.0, 0.2),
        (100.0, 0.0, 1.0, 0.2),
        (100.0, 100.0, -1.0, 0.2),
        (100.0, 100.0, 1.0, 0.0),
    ],
)
def test_rejects_bad_inputs(spot: float, strike: float, t: float, vol: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        bs_price(spot, strike, t, vol, OptionRight.call)
