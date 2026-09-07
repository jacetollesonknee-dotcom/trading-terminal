"""The object ``run_backtest`` returns: ledger + config + metrics.

Kept in its own module (rather than in ``engine`` or ``metrics``) so both can
import it without a cycle: ``engine`` builds a result, ``metrics`` is called by
the result to populate itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.config import BacktestConfig
from backtest.metrics import PerformanceMetrics, compute_metrics


@dataclass(frozen=True)
class BacktestResult:
    """Everything a run produced, in one immutable object.

    Attributes:
        ledger: Per-bar DataFrame — ``returns, position, turnover, gross,
            funding, costs, net, equity``.
        config: The :class:`BacktestConfig` the run used.
        metrics: Summary :class:`PerformanceMetrics`.
    """

    ledger: pd.DataFrame
    config: BacktestConfig
    metrics: PerformanceMetrics

    @classmethod
    def from_ledger(cls, ledger: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
        """Build a result, computing metrics from the ledger."""
        return cls(ledger=ledger, config=config, metrics=compute_metrics(ledger, config))

    @property
    def equity_curve(self) -> pd.Series:
        """The equity curve, indexed by timestamp."""
        return self.ledger["equity"]

    def summary(self) -> dict[str, float | int]:
        """Flat dict of headline metrics — convenient for logging / reports."""
        return self.metrics.as_dict()
