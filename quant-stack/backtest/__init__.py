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
from backtest.engine import run_backtest
from backtest.metrics import PerformanceMetrics, compute_metrics
from backtest.result import BacktestResult

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "PerformanceMetrics",
    "compute_metrics",
    "run_backtest",
]
