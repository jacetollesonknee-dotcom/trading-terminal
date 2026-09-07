"""Backtesting and validation toolkit.

The vectorized engine (:func:`run_backtest`) prices a linear position in an
underlying; the options simulator in :mod:`options.simulator` prices a
naked-only options book over chain snapshots. Both produce the same ledger
shape, so the validation tools here — metrics, deflated Sharpe, walk-forward,
regime attribution, sizing, health monitor — apply to either.

    >>> from backtest import BacktestConfig, run_backtest
    >>> result = run_backtest(prices, signal, BacktestConfig())
    >>> result.metrics.sharpe
"""

from __future__ import annotations

from backtest.config import BacktestConfig
from backtest.deflated_sharpe import (
    DeflatedSharpeResult,
    deannualize_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    minimum_sharpe_to_pass,
    probabilistic_sharpe,
)
from backtest.engine import run_backtest, validate_prices
from backtest.metrics import PerformanceMetrics, compute_metrics
from backtest.monitor import (
    Benchmark,
    HealthConfig,
    HealthReport,
    health_check,
    longest_underwater,
)
from backtest.regimes import (
    Regime,
    RegimeConfig,
    RegimeReport,
    RegimeStats,
    classify_regimes,
    regime_report,
)
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
    "Benchmark",
    "DeflatedSharpeResult",
    "FitFn",
    "FittedStrategy",
    "Fold",
    "HealthConfig",
    "HealthReport",
    "PerformanceMetrics",
    "PositionSize",
    "Regime",
    "RegimeConfig",
    "RegimeReport",
    "RegimeStats",
    "Side",
    "SizingConfig",
    "WalkForwardConfig",
    "WalkForwardResult",
    "classify_regimes",
    "compute_metrics",
    "deannualize_sharpe",
    "deflated_sharpe",
    "expected_max_sharpe",
    "health_check",
    "longest_underwater",
    "minimum_sharpe_to_pass",
    "position_size",
    "probabilistic_sharpe",
    "regime_report",
    "run_backtest",
    "validate_prices",
    "walk_forward",
]
