"""Walk-forward validation: fit on the past, trade the unseen future, roll.

A strategy fitted on the whole dataset and then backtested on it tells you
how well it fits, not how well it trades. Walk-forward splits time into
rolling (or anchored) train/test folds, fits on each train window only, and
evaluates on the test window that follows — the closest a backtest gets to
the experience of actually running the thing.

Two corrections to the textbook loop, both of which distort every fold:

1. **The signal sees contiguous history, not the bare test slice.** A lookback
   indicator (moving average, z-score, ...) evaluated on a 60-bar test slice
   alone spends its first ``lookback`` bars warming up — with a 50-bar
   lookback that is 49 of 60 bars wasted, per fold. Here the fitted signal
   function receives the price history running *through* the test window.
   Looking at the past is fair game; the engine's ``execution_lag`` shift is
   what guards the future.

2. **One stitched out-of-sample backtest, not one backtest per fold.**
   Re-running the engine on each fold restarts the position at zero every
   time: the first bar of every fold is a phantom flat bar, the previous
   fold's closing position is dropped without an exit cost, and a fresh
   entry cost is charged. Here the out-of-sample signals are stitched into
   one contiguous series and backtested *once*, so positions carry across
   fold boundaries exactly as they would have in live trading. The headline
   Sharpe is that of the stitched series; per-fold statistics are
   consistency *diagnostics* (a 60-bar Sharpe is far too noisy to be a
   result on its own).

The driver never lets ``fit_fn`` see a test bar. It cannot, however, see
inside the closure it is handed: a ``signal_fn`` that refits on the data it
is given would leak, and that is the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig
from backtest.engine import run_backtest, validate_prices
from backtest.metrics import compute_metrics
from backtest.result import BacktestResult

# A train window needs at least this many bars to fit anything meaningful.
_MIN_TRAIN_BARS = 2


@dataclass(frozen=True)
class FittedStrategy:
    """What ``fit_fn`` returns for one train window.

    Attributes:
        signal_fn: Maps a price history to a target-position series on the
            SAME index (``[-max_leverage, +max_leverage]``, ``NaN`` = flat).
            It is handed history running through the test window; it may use
            the past freely and must not peek forward within the series.
        params: The fitted parameters, recorded per fold so their stability
            across folds can be inspected — parameters that jump wildly
            fold-to-fold are a classic overfitting tell.
    """

    signal_fn: Callable[[pd.Series], pd.Series]
    params: Mapping[str, float] = field(default_factory=dict)


FitFn = Callable[[pd.Series], FittedStrategy]


@dataclass(frozen=True)
class WalkForwardConfig:
    """Fold geometry.

    Attributes:
        train_bars: Bars in each train window. Named in *bars*, not days —
            the split is positional and knows nothing about the bar interval.
        test_bars: Bars in each test window; also the roll step.
        anchored: If True the train window grows from the first bar
            (expanding); if False it rolls with a fixed length.
    """

    train_bars: int = 180
    test_bars: int = 60
    anchored: bool = False

    def __post_init__(self) -> None:
        if self.train_bars < _MIN_TRAIN_BARS:
            msg = f"train_bars must be >= {_MIN_TRAIN_BARS}, got {self.train_bars}"
            raise ValueError(msg)
        if self.test_bars < 1:
            msg = f"test_bars must be >= 1, got {self.test_bars}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Fold:
    """One train/test split, with the parameters fitted on it."""

    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: Mapping[str, float]


@dataclass(frozen=True)
class WalkForwardResult:
    """Outcome of a walk-forward run.

    Attributes:
        result: ONE :class:`BacktestResult` over the stitched out-of-sample
            period. Its metrics are the headline numbers.
        folds: The splits, in order, each carrying its fitted params.
        fold_metrics: Per-fold diagnostics (Sharpe, return, drawdown, ...)
            computed from the stitched ledger sliced per fold and re-based to
            ``initial_capital``. Consistency checks, not results.
        param_table: Fitted params per fold, one row per fold — eyeball this
            for stability.
    """

    result: BacktestResult
    folds: tuple[Fold, ...]
    fold_metrics: pd.DataFrame
    param_table: pd.DataFrame

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def positive_folds(self) -> int:
        """Folds whose local Sharpe is > 0 (a NaN Sharpe — flat fold — counts as not positive)."""
        return int((self.fold_metrics["sharpe"] > 0).sum())

    def summary(self) -> dict[str, float | int]:
        """Headline OOS metrics plus fold-consistency diagnostics."""
        m = self.result.metrics
        return {
            "oos_sharpe": m.sharpe,
            "oos_cagr": m.cagr,
            "oos_max_drawdown": m.max_drawdown,
            "oos_total_return": m.total_return,
            "n_folds": self.n_folds,
            "positive_folds": self.positive_folds,
            "mean_fold_sharpe": float(self.fold_metrics["sharpe"].mean()),
            "worst_fold_sharpe": float(self.fold_metrics["sharpe"].min()),
        }


def _fold_local_metrics(ledger: pd.DataFrame, cfg: BacktestConfig) -> dict[str, float | int]:
    """Metrics for one fold's slice of the stitched ledger, re-based to initial capital.

    The stitched equity curve mid-run is not ``initial_capital``, so the
    slice's equity is recomputed from its own ``net`` before scoring.
    """
    local = ledger.copy()
    local["equity"] = cfg.initial_capital * np.exp(local["net"].cumsum())
    m = compute_metrics(local, cfg)
    return {
        "n_bars": m.n_periods,
        "sharpe": m.sharpe,
        "total_return": m.total_return,
        "max_drawdown": m.max_drawdown,
        "num_trades": m.num_trades,
        "exposure": m.exposure,
    }


def walk_forward(
    prices: pd.Series,
    fit_fn: FitFn,
    cfg: BacktestConfig | None = None,
    wf: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Run a walk-forward validation.

    Args:
        prices: Close prices, ascending unique ``DatetimeIndex``, strictly
            positive — the same contract as :func:`run_backtest`.
        fit_fn: ``fit_fn(train_prices) -> FittedStrategy``. Receives ONLY the
            train window.
        cfg: Backtest economics. Defaults to :class:`BacktestConfig`.
        wf: Fold geometry. Defaults to :class:`WalkForwardConfig`.

    Returns:
        A :class:`WalkForwardResult`.

    Raises:
        ValueError: If there aren't enough bars for a single fold, or a
            ``signal_fn`` returns a series on a different index than it was
            given (reindexing would silently fabricate or drop bars).

    Note:
        Trailing bars that don't fill a whole test window are not traded;
        the stitched result spans exactly ``n_folds * test_bars`` bars.
    """
    if cfg is None:
        cfg = BacktestConfig()
    if wf is None:
        wf = WalkForwardConfig()
    validate_prices(prices)

    n = len(prices)
    if n < wf.train_bars + wf.test_bars:
        msg = (
            f"need >= train_bars + test_bars = {wf.train_bars + wf.test_bars} bars "
            f"for one fold, got {n}"
        )
        raise ValueError(msg)

    oos_signal = pd.Series(np.nan, index=prices.index, dtype=float)
    folds: list[Fold] = []
    i = 0
    while i + wf.train_bars + wf.test_bars <= n:
        train_lo = 0 if wf.anchored else i
        train_hi = i + wf.train_bars
        test_hi = train_hi + wf.test_bars

        train = prices.iloc[train_lo:train_hi]
        fitted = fit_fn(train)

        # History runs THROUGH the test window: the signal may use everything
        # up to each bar (the engine's shift keeps it from acting on that bar),
        # so lookback indicators are already warm when the test window begins.
        history = prices.iloc[train_lo:test_hi]
        signal = fitted.signal_fn(history)
        if not isinstance(signal, pd.Series) or not signal.index.equals(history.index):
            msg = (
                "signal_fn must return a Series on the same index it was given "
                f"(fold {len(folds)})"
            )
            raise ValueError(msg)

        test_index = prices.index[train_hi:test_hi]
        oos_signal.loc[test_index] = signal.loc[test_index].to_numpy(dtype=float)

        folds.append(
            Fold(
                number=len(folds),
                train_start=train.index[0],
                train_end=train.index[-1],
                test_start=test_index[0],
                test_end=test_index[-1],
                params=dict(fitted.params),
            )
        )
        i += wf.test_bars

    # One backtest over the contiguous out-of-sample span. Positions carry
    # across fold boundaries; the only cold start is the very first OOS bar.
    oos_lo = wf.train_bars
    oos_hi = wf.train_bars + len(folds) * wf.test_bars
    result = run_backtest(prices.iloc[oos_lo:oos_hi], oos_signal.iloc[oos_lo:oos_hi], cfg)

    fold_rows = []
    for fold in folds:
        piece = result.ledger.loc[fold.test_start : fold.test_end]
        fold_rows.append(
            {
                "fold": fold.number,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **_fold_local_metrics(piece, cfg),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows).set_index("fold")
    param_table = pd.DataFrame([dict(f.params) for f in folds], index=[f.number for f in folds])
    param_table.index.name = "fold"

    return WalkForwardResult(
        result=result,
        folds=tuple(folds),
        fold_metrics=fold_metrics,
        param_table=param_table,
    )
