"""Vectorized crypto backtesting.

A lightweight, look-ahead-safe backtester for target-position signals on a
single crypto instrument. Feed it close prices and a per-bar target position;
it returns an equity curve, a cost-aware P&L ledger, and performance metrics.

    >>> from backtest import BacktestConfig, run_backtest
    >>> result = run_backtest(prices, signal, BacktestConfig())
    >>> result.metrics.sharpe

This is the *vectorized* backtester for fast signal research on crypto (24/7,
no options, no calendar). It is distinct from the Phase 2 event-driven,
options-aware backtester described in the project brief.
"""

from __future__ import annotations

from backtest.config import BacktestConfig
from backtest.deflated_sharpe import (
    DeflatedSharpeResult,
    deannualize_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)
from backtest.engine import run_backtest, validate_prices
from backtest.metrics import PerformanceMetrics, compute_metrics
from backtest.result import BacktestResult
from backtest.sizing import PositionSize, Side, SizingConfig, position_size
from backtest.walk_forward import (
    FitFn,
    FittedStrategy,
    Fold,
    WalkForwardConfig,
    WalkForwardResult,
    walk_forward,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "DeflatedSharpeResult",
    "FitFn",
    "FittedStrategy",
    "Fold",
    "PerformanceMetrics",
    "PositionSize",
    "Side",
    "SizingConfig",
    "WalkForwardConfig",
    "WalkForwardResult",
    "compute_metrics",
    "deannualize_sharpe",
    "deflated_sharpe",
    "expected_max_sharpe",
    "position_size",
    "probabilistic_sharpe",
    "run_backtest",
    "validate_prices",
    "walk_forward",
]
