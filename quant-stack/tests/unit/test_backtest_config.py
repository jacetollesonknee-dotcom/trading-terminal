"""BacktestConfig validation tests."""

from __future__ import annotations

import pytest

from backtest import BacktestConfig


def test_defaults_are_daily_equity() -> None:
    cfg = BacktestConfig()
    assert cfg.periods_per_year == 252
    assert cfg.execution_lag == 1
    assert cfg.slippage_bps > 0  # spreads aren't free
    assert cfg.carry_bps_per_year == 0.0


def test_cost_and_carry_rate_derivation() -> None:
    cfg = BacktestConfig(
        fee_bps=5.0, slippage_bps=3.0, carry_bps_per_year=252.0, periods_per_year=252
    )
    assert cfg.cost_rate == pytest.approx(8.0 / 1e4)
    # 252 bps/yr over 252 daily periods = 1 bp/day.
    assert cfg.carry_rate_per_period == pytest.approx(1.0 / 1e4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_capital": 0},
        {"initial_capital": -1},
        {"fee_bps": -0.1},
        {"slippage_bps": -0.1},
        {"max_leverage": 0},
        {"max_leverage": -1},
        {"periods_per_year": 0},
        {"execution_lag": 0},  # 0 would be look-ahead
        {"execution_lag": -1},
        {"carry_bps_per_year": -1},
    ],
)
def test_invalid_configs_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        BacktestConfig(**kwargs)  # type: ignore[arg-type]


def test_frozen() -> None:
    cfg = BacktestConfig()
    with pytest.raises(AttributeError):
        cfg.fee_bps = 10.0  # type: ignore[misc]
