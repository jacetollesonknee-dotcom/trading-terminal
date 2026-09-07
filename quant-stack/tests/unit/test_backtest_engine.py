"""Engine behavior + input-validation tests for the vectorized backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestConfig, run_backtest


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")


def _prices(values: list[float]) -> pd.Series:
    return pd.Series(values, index=_index(len(values)), name="close")


# ─────────────────────────────────────────────────────────────────────────────
#  Core mechanics
# ─────────────────────────────────────────────────────────────────────────────


def test_flat_signal_keeps_capital_flat() -> None:
    prices = _prices([100, 110, 90, 105, 120])
    signal = pd.Series(0.0, index=prices.index)
    result = run_backtest(prices, signal, BacktestConfig())
    # Never in the market -> equity never moves, no costs.
    assert result.equity_curve.iloc[-1] == pytest.approx(10_000.0)
    assert result.metrics.total_return == pytest.approx(0.0)
    assert result.metrics.total_turnover == pytest.approx(0.0)


def test_full_long_matches_buy_and_hold_before_costs() -> None:
    prices = _prices([100, 110, 121, 133.1])
    signal = pd.Series(1.0, index=prices.index)
    # Zero costs so gross == net; a lag-1 always-long position is invested from
    # bar 1 onward, capturing every return except the (nonexistent) bar-0 one.
    cfg = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    result = run_backtest(prices, signal, cfg)
    expected = 10_000.0 * (prices.iloc[-1] / prices.iloc[0])
    assert result.equity_curve.iloc[-1] == pytest.approx(expected)


def test_costs_charged_on_turnover() -> None:
    prices = _prices([100, 100, 100, 100])  # flat prices -> only costs move equity
    signal = pd.Series([0.0, 1.0, 0.0, 0.0], index=prices.index)
    cfg = BacktestConfig(fee_bps=10.0, slippage_bps=0.0)
    result = run_backtest(prices, signal, cfg)
    # Position (lag 1): [0, 0, 1, 0]. Turnover: enter (0->1)=1 then exit (1->0)=1.
    assert result.metrics.total_turnover == pytest.approx(2.0)
    # Two units of turnover at 10 bps each = 20 bps total drag on flat prices.
    assert result.equity_curve.iloc[-1] == pytest.approx(10_000.0 * np.exp(-2 * 10.0 / 1e4))


def test_leverage_cap_applied() -> None:
    prices = _prices([100, 110, 120])
    signal = pd.Series(5.0, index=prices.index)  # way over the cap
    cfg = BacktestConfig(max_leverage=1.0, fee_bps=0.0, slippage_bps=0.0)
    result = run_backtest(prices, signal, cfg)
    assert (result.ledger["position"].abs() <= 1.0 + 1e-12).all()


def test_short_profits_when_price_falls() -> None:
    prices = _prices([100, 90, 81])
    signal = pd.Series(-1.0, index=prices.index)
    cfg = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    result = run_backtest(prices, signal, cfg)
    assert result.equity_curve.iloc[-1] > 10_000.0


def test_carry_charged_on_held_position() -> None:
    prices = _prices([100, 100, 100])
    signal = pd.Series(1.0, index=prices.index)
    cfg = BacktestConfig(fee_bps=0.0, slippage_bps=0.0, carry_bps_per_year=252.0)
    result = run_backtest(prices, signal, cfg)
    # Flat prices, but a held long pays carry once it's on (bars 1 and 2).
    assert result.equity_curve.iloc[-1] < 10_000.0
    assert result.ledger["carry"].sum() > 0.0


def test_nan_signal_treated_as_flat() -> None:
    prices = _prices([100, 110, 120, 130])
    signal = pd.Series([np.nan, np.nan, 1.0, 1.0], index=prices.index)
    result = run_backtest(prices, signal, BacktestConfig(fee_bps=0.0, slippage_bps=0.0))
    # Warm-up NaNs -> flat, so position stays 0 until the lagged real signal.
    assert result.ledger["position"].iloc[0] == 0.0
    assert result.ledger["position"].iloc[1] == 0.0


def test_default_config_used_when_omitted() -> None:
    prices = _prices([100, 105, 110])
    signal = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signal)
    assert result.config == BacktestConfig()


# ─────────────────────────────────────────────────────────────────────────────
#  Input validation — no silent fallbacks
# ─────────────────────────────────────────────────────────────────────────────


def test_rejects_misaligned_index() -> None:
    prices = _prices([100, 110, 120])
    signal = pd.Series(1.0, index=_index(3) + pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="same index"):
        run_backtest(prices, signal)


def test_rejects_non_datetime_index() -> None:
    prices = pd.Series([100.0, 110.0, 120.0], index=[0, 1, 2])
    signal = pd.Series([1.0, 1.0, 1.0], index=[0, 1, 2])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        run_backtest(prices, signal)


def test_rejects_non_positive_prices() -> None:
    prices = _prices([100, 0, 120])
    signal = pd.Series(1.0, index=prices.index)
    with pytest.raises(ValueError, match="strictly positive"):
        run_backtest(prices, signal)


def test_rejects_nan_prices() -> None:
    prices = _prices([100, np.nan, 120])
    signal = pd.Series(1.0, index=prices.index)
    with pytest.raises(ValueError, match="NaN or inf"):
        run_backtest(prices, signal)


def test_rejects_unsorted_index() -> None:
    idx = pd.DatetimeIndex(
        ["2024-01-03", "2024-01-01", "2024-01-02"], tz="UTC"
    )
    prices = pd.Series([100.0, 110.0, 120.0], index=idx)
    signal = pd.Series(1.0, index=idx)
    with pytest.raises(ValueError, match="sorted ascending"):
        run_backtest(prices, signal)


def test_rejects_too_short() -> None:
    prices = _prices([100])
    signal = pd.Series(1.0, index=prices.index)
    with pytest.raises(ValueError, match=">= 2"):
        run_backtest(prices, signal)


def test_rejects_non_series() -> None:
    with pytest.raises(TypeError):
        run_backtest([100, 110, 120], pd.Series([1.0]))
