"""Vectorized crypto backtester — the no-look-ahead core.

One function, :func:`run_backtest`, turns a price series and a target-position
signal into a per-bar P&L ledger and an equity curve. It is deliberately
vectorized (pandas, no Python loop over bars) so a multi-year daily backtest
runs in milliseconds.

The single most important line in this module is the ``.shift(cfg.execution_lag)``
that turns a *signal* (computed from data available up to and including bar *t*)
into a *position* held from bar *t + lag* onward. You act on the next bar, not
the one you just saw. :class:`~backtest.config.BacktestConfig` refuses a lag
below 1, so this guarantee cannot be configured away.

Costs are charged on turnover (``|Δposition|``) in basis points of notional,
plus an optional per-bar funding carry on the absolute position for
perpetual-swap strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig
from backtest.result import BacktestResult

# Columns of the returned ledger, in order.
_COLUMNS = ("returns", "position", "turnover", "gross", "funding", "costs", "net", "equity")

# Minimum bars needed to compute a single return.
_MIN_OBSERVATIONS = 2


def validate_prices(prices: pd.Series) -> None:
    """Reject a price series that would silently corrupt a backtest.

    Shared by :func:`run_backtest` and the walk-forward driver. Non-monotonic
    time, duplicate timestamps, and non-positive prices are hard errors, not
    quietly-repaired warnings.
    """
    if not isinstance(prices, pd.Series):
        msg = f"prices must be a pandas Series, got {type(prices).__name__}"
        raise TypeError(msg)
    if len(prices) < _MIN_OBSERVATIONS:
        msg = (
            f"prices needs >= {_MIN_OBSERVATIONS} observations to compute a return, "
            f"got {len(prices)}"
        )
        raise ValueError(msg)
    if not isinstance(prices.index, pd.DatetimeIndex):
        msg = "prices/signal must be indexed by a DatetimeIndex"
        raise TypeError(msg)
    if not prices.index.is_monotonic_increasing:
        msg = "index must be sorted ascending in time"
        raise ValueError(msg)
    if not prices.index.is_unique:
        msg = "index has duplicate timestamps"
        raise ValueError(msg)
    if not np.isfinite(prices.to_numpy()).all():
        msg = "prices contains NaN or inf"
        raise ValueError(msg)
    if (prices <= 0).any():
        # Log returns are undefined for non-positive prices.
        msg = "prices must be strictly positive"
        raise ValueError(msg)


def _validate_inputs(prices: pd.Series, signal: pd.Series) -> None:
    """Validate the price series, then the signal against it.

    ``NaN`` in the *signal* is the one tolerated gap — it is treated as "no
    position" (flat), which is a legitimate warm-up state. A misaligned index
    is not tolerated: reindexing would fabricate or drop bars silently.
    """
    validate_prices(prices)
    if not isinstance(signal, pd.Series):
        msg = f"signal must be a pandas Series, got {type(signal).__name__}"
        raise TypeError(msg)
    if not prices.index.equals(signal.index):
        msg = "prices and signal must share the exact same index"
        raise ValueError(msg)


def run_backtest(
    prices: pd.Series,
    signal: pd.Series,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest a target-position signal against a price series.

    Args:
        prices: Close prices indexed by an ascending, unique ``DatetimeIndex``.
            Must be strictly positive.
        signal: Target position in ``[-max_leverage, +max_leverage]`` at each
            timestamp, computed from data available AT that timestamp. Same
            index as ``prices``. ``NaN`` is treated as flat (no position).
        cfg: Economic assumptions. Defaults to :class:`BacktestConfig` if omitted.

    Returns:
        A :class:`BacktestResult` wrapping the per-bar ledger, the resolved
        config, and computed performance metrics.
    """
    if cfg is None:
        cfg = BacktestConfig()

    _validate_inputs(prices, signal)

    # THE no-look-ahead line. A signal seen at bar t is acted on at t + lag.
    # NaN (warm-up) becomes flat; the position is then capped to leverage.
    position = signal.shift(cfg.execution_lag).fillna(0.0).clip(-cfg.max_leverage, cfg.max_leverage)

    # Log returns of the asset. The leading NaN is structural (no prior bar);
    # the position there is 0 anyway, so it contributes nothing.
    returns = np.log(prices / prices.shift(1)).fillna(0.0)
    gross = position * returns

    # Turnover drives transaction costs; the leading diff is NaN -> 0.
    turnover = position.diff().abs().fillna(0.0)
    costs = turnover * cfg.cost_rate

    # Optional perpetual-swap funding / borrow carry on the held notional.
    funding = position.abs() * cfg.funding_rate_per_period

    net = gross - costs - funding
    equity = cfg.initial_capital * np.exp(net.cumsum())

    ledger = pd.DataFrame(
        {
            "returns": returns,
            "position": position,
            "turnover": turnover,
            "gross": gross,
            "funding": funding,
            "costs": costs,
            "net": net,
            "equity": equity,
        },
        columns=list(_COLUMNS),
    )
    return BacktestResult.from_ledger(ledger, cfg)
