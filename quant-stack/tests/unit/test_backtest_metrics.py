"""Performance-metrics tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest import BacktestConfig, run_backtest


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")


def test_max_drawdown_is_negative_on_a_dip() -> None:
    prices = pd.Series([100, 120, 60, 90], index=_index(4), dtype=float)
    signal = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signal, BacktestConfig(fee_bps=0.0, slippage_bps=0.0))
    assert result.metrics.max_drawdown < 0.0


def test_flat_run_has_undefined_ratios() -> None:
    prices = pd.Series([100, 100, 100, 100], index=_index(4), dtype=float)
    signal = pd.Series(0.0, index=prices.index)
    m = run_backtest(prices, signal, BacktestConfig()).metrics
    # No volatility, no drawdown, no active bars -> ratios are nan, not fake zeros.
    assert m.ann_volatility == 0.0
    assert math.isnan(m.sharpe)
    assert math.isnan(m.sortino)
    assert math.isnan(m.hit_rate)
    assert m.exposure == 0.0


def test_positive_trend_gives_positive_sharpe() -> None:
    prices = pd.Series(100 * np.exp(np.linspace(0, 0.5, 60)), index=_index(60))
    # Small deterministic wiggle so volatility (and thus Sharpe) is defined.
    prices = prices * (1 + 0.001 * np.sin(np.arange(60)))
    signal = pd.Series(1.0, index=prices.index)
    m = run_backtest(prices, signal, BacktestConfig(fee_bps=0.0, slippage_bps=0.0)).metrics
    assert m.total_return > 0.0
    assert m.cagr > 0.0
    assert m.sharpe > 0.0


def test_summary_dict_has_all_fields() -> None:
    prices = pd.Series([100, 110, 105, 115], index=_index(4), dtype=float)
    signal = pd.Series(1.0, index=prices.index)
    summary = run_backtest(prices, signal).summary()
    for key in ("sharpe", "cagr", "max_drawdown", "num_trades", "final_equity"):
        assert key in summary
