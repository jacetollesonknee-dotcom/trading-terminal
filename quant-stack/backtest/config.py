"""Configuration for the vectorized crypto backtester.

Crypto-specific defaults: markets trade 24/7 so a "year" is 365 periods for
daily bars, spreads are wider than equities (hence a non-trivial slippage
default), and there is no exchange holiday calendar to reason about.

The config is a frozen dataclass — construct once, never mutate. All economic
assumptions a run depends on live here so a result can be reproduced from
``(prices, signal, BacktestConfig)`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    """Economic assumptions for one backtest run.

    Attributes:
        initial_capital: Starting equity in quote currency (e.g. USD).
        fee_bps: Exchange fee charged per unit of turnover, in basis points of
            notional. Turnover is ``|Δposition|``, so a full round trip
            (0 → 1 → 0) is charged twice — once entering, once exiting.
        slippage_bps: Modelled execution slippage per unit of turnover, in
            basis points. Crypto spreads are wider than equities; do not set
            this to zero for a realistic run.
        max_leverage: Absolute cap applied to the target position after the
            execution lag. ``1.0`` means fully invested, never levered.
        periods_per_year: Bars per year, used to annualize returns and
            volatility. 365 for daily crypto bars; 365*24 for hourly, etc.
        execution_lag: Number of bars between a signal being *observed* and the
            position being *held*. MUST be >= 1: this is the no-look-ahead
            guarantee — you act on the next bar, never the one you just saw.
        funding_bps_per_year: Annualized carry charged on the absolute position
            each bar, in basis points. Models perpetual-swap funding / borrow
            cost. Default 0.0 (spot, no carry).
        risk_free_rate: Annualized risk-free rate (as a decimal, log-space) used
            as the benchmark in Sharpe / Sortino. Default 0.0.
    """

    initial_capital: float = 10_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 3.0
    max_leverage: float = 1.0
    periods_per_year: int = 365
    execution_lag: int = 1
    funding_bps_per_year: float = 0.0
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            msg = f"initial_capital must be > 0, got {self.initial_capital}"
            raise ValueError(msg)
        if self.fee_bps < 0:
            msg = f"fee_bps must be >= 0, got {self.fee_bps}"
            raise ValueError(msg)
        if self.slippage_bps < 0:
            msg = f"slippage_bps must be >= 0, got {self.slippage_bps}"
            raise ValueError(msg)
        if self.max_leverage <= 0:
            msg = f"max_leverage must be > 0, got {self.max_leverage}"
            raise ValueError(msg)
        if self.periods_per_year <= 0:
            msg = f"periods_per_year must be > 0, got {self.periods_per_year}"
            raise ValueError(msg)
        if self.execution_lag < 1:
            # A lag of 0 would let the position react to the very bar whose
            # close produced the signal — the canonical look-ahead bug.
            msg = f"execution_lag must be >= 1 (no look-ahead), got {self.execution_lag}"
            raise ValueError(msg)
        if self.funding_bps_per_year < 0:
            msg = f"funding_bps_per_year must be >= 0, got {self.funding_bps_per_year}"
            raise ValueError(msg)

    @property
    def cost_rate(self) -> float:
        """Per-unit-turnover cost as a fraction of notional (fee + slippage)."""
        return (self.fee_bps + self.slippage_bps) / 1e4

    @property
    def funding_rate_per_period(self) -> float:
        """Funding/borrow charged per bar, as a fraction of absolute position."""
        return (self.funding_bps_per_year / 1e4) / self.periods_per_year
