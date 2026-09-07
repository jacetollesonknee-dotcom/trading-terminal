"""Position-sizing tests."""

from __future__ import annotations

import pytest

from backtest import PositionSize, Side, SizingConfig, position_size


def _free() -> SizingConfig:
    """No slippage, no fees — isolates the pure fixed-fractional math."""
    return SizingConfig(stop_slippage_bps=0.0, fee_bps=0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Core math
# ─────────────────────────────────────────────────────────────────────────────


def test_basic_long_risks_exactly_the_budget() -> None:
    # $10k, 1% risk = $100. Stop $5 away -> 20 units, $2000 notional (20%).
    ps = position_size(10_000, 100, 95, Side.long, _free())
    assert ps.units == pytest.approx(20.0)
    assert ps.notional == pytest.approx(2_000.0)
    assert ps.position_fraction == pytest.approx(0.20)
    assert ps.loss_if_stopped == pytest.approx(100.0)
    assert ps.risk_fraction_realized == pytest.approx(0.01)
    assert not ps.capped and not ps.floored and not ps.below_min_notional


def test_short_is_negative_fraction_same_magnitude() -> None:
    ps = position_size(10_000, 100, 105, Side.short, _free())
    assert ps.units == pytest.approx(20.0)
    assert ps.position_fraction == pytest.approx(-0.20)
    assert ps.loss_if_stopped == pytest.approx(100.0)


def test_cap_binds_on_tight_stop_and_risk_drops_below_target() -> None:
    # Stop $1 away wants 100 units = $10k notional; cap is 20% = $2k -> 20 units.
    ps = position_size(10_000, 100, 99, Side.long, _free())
    assert ps.capped
    assert ps.units == pytest.approx(20.0)
    assert ps.position_fraction == pytest.approx(0.20)
    # The visible consequence: you're risking 0.2%, not the 1% you asked for.
    assert ps.loss_if_stopped == pytest.approx(20.0)
    assert ps.risk_fraction_realized == pytest.approx(0.002)


def test_position_fraction_never_exceeds_cap() -> None:
    cfg = SizingConfig(max_position_fraction=0.5, stop_slippage_bps=0.0, fee_bps=0.0)
    for stop in (99.99, 99.0, 90.0, 50.0, 1.0):
        ps = position_size(10_000, 100, stop, Side.long, cfg)
        assert abs(ps.position_fraction) <= 0.5 + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
#  Honest stop: slippage + fees are inside the budget
# ─────────────────────────────────────────────────────────────────────────────


def test_costs_shrink_size_so_realized_loss_still_equals_budget() -> None:
    naive = position_size(10_000, 100, 95, Side.long, _free())
    with_costs = position_size(
        10_000, 100, 95, Side.long, SizingConfig(stop_slippage_bps=10.0, fee_bps=10.0)
    )
    # Fewer units once the stop is assumed to fill worse and fees are paid...
    assert with_costs.units < naive.units
    # ...but the all-in loss if stopped is still exactly the 1% budget.
    assert with_costs.loss_if_stopped == pytest.approx(100.0)
    assert with_costs.risk_fraction_realized == pytest.approx(0.01)


def test_naive_size_would_overshoot_budget_once_costs_are_counted() -> None:
    """What the textbook version gets wrong, stated as a number."""
    naive = position_size(10_000, 100, 95, Side.long, _free())
    slip_bps, fee_bps = 10.0, 10.0
    # Real loss on the naive 20 units: distance + slippage on stop + fees both legs.
    real_loss = naive.units * (5.0 + 95 * slip_bps / 1e4 + (100 + 95) * fee_bps / 1e4)
    assert real_loss > 100.0  # exceeds the 1% budget


def test_float_noise_stop_is_bounded_by_slippage_not_exploded() -> None:
    # Stop 1e-9 away. Without a slippage floor, units = $100 / 1e-9 before the
    # cap rescues it and loss_if_stopped reads 0.0 — a sure stop-out reported
    # as riskless. With slippage, effective risk/unit >= slippage, so the
    # loss is honest and non-zero.
    ps = position_size(10_000, 100, 100 - 1e-9, Side.long, SizingConfig(stop_slippage_bps=3.0))
    assert ps.capped
    assert ps.loss_if_stopped > 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Venue constraints
# ─────────────────────────────────────────────────────────────────────────────


def test_lot_step_floors_never_rounds_up() -> None:
    # 20.7 units wanted; step 0.5 -> 20.5, never 21.0.
    cfg = SizingConfig(stop_slippage_bps=0.0, fee_bps=0.0, lot_step=0.5)
    ps = position_size(10_350, 100, 95, Side.long, cfg)  # 103.5/5 = 20.7
    assert ps.units == pytest.approx(20.5)
    assert ps.floored
    assert ps.risk_fraction_realized < 0.01


def test_lot_step_no_op_when_already_aligned() -> None:
    cfg = SizingConfig(stop_slippage_bps=0.0, fee_bps=0.0, lot_step=0.5)
    ps = position_size(10_000, 100, 95, Side.long, cfg)  # exactly 20.0
    assert ps.units == pytest.approx(20.0)
    assert not ps.floored


def test_min_notional_zeroes_untradeable_size() -> None:
    cfg = SizingConfig(stop_slippage_bps=0.0, fee_bps=0.0, min_notional=5_000.0)
    ps = position_size(10_000, 100, 95, Side.long, cfg)  # $2000 < $5000 minimum
    assert ps.below_min_notional
    assert ps.units == 0.0
    assert ps.notional == 0.0
    assert ps.position_fraction == 0.0
    assert ps.loss_if_stopped == 0.0


def test_result_is_immutable() -> None:
    ps = position_size(10_000, 100, 95, Side.long, _free())
    assert isinstance(ps, PositionSize)
    with pytest.raises(AttributeError):
        ps.units = 1.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
#  Validation — no silent fallbacks
# ─────────────────────────────────────────────────────────────────────────────


def test_rejects_long_with_stop_above_entry() -> None:
    with pytest.raises(ValueError, match="long stop must be below"):
        position_size(10_000, 100, 105, Side.long)


def test_rejects_short_with_stop_below_entry() -> None:
    with pytest.raises(ValueError, match="short stop must be above"):
        position_size(10_000, 100, 95, Side.short)


def test_rejects_stop_equal_to_entry() -> None:
    with pytest.raises(ValueError, match="cannot equal"):
        position_size(10_000, 100, 100, Side.long)


@pytest.mark.parametrize("capital", [0.0, -10_000.0, float("nan"), float("inf")])
def test_rejects_bad_capital(capital: float) -> None:
    with pytest.raises(ValueError, match="capital"):
        position_size(capital, 100, 95, Side.long)


@pytest.mark.parametrize("entry", [0.0, -100.0, float("nan")])
def test_rejects_bad_entry(entry: float) -> None:
    with pytest.raises(ValueError, match="entry"):
        position_size(10_000, entry, 95, Side.long)


@pytest.mark.parametrize("stop", [0.0, -5.0, float("inf")])
def test_rejects_bad_stop(stop: float) -> None:
    with pytest.raises(ValueError, match="stop"):
        position_size(10_000, 100, stop, Side.long)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_fraction": 0.0},
        {"risk_fraction": 1.5},
        {"risk_fraction": -0.01},
        {"max_position_fraction": 0.0},
        {"stop_slippage_bps": -1.0},
        {"fee_bps": -1.0},
        {"lot_step": 0.0},
        {"min_notional": -1.0},
    ],
)
def test_rejects_bad_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        SizingConfig(**kwargs)
