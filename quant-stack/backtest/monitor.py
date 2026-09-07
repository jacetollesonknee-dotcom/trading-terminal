"""Live health monitoring: is the strategy still the one you backtested?

Run this on a schedule. It **recommends**; it never acts. Nothing in this
module can touch an order, and its output is a report for the person who
presses the buttons. That is a design decision, not a missing feature.

What the textbook version gets wrong, and what this does instead:

- **A 30-bar Sharpe is noise.** Its standard error is ``sqrt(periods_per_year
  / window)`` — about 3.5 for daily bars. A rule like "halt if the live
  Sharpe drops below half the backtest's" fires on ~40% of days for a
  strategy with a *true* Sharpe of 1.5. This module asks a better question:
  "how likely is a window this bad if the strategy still has its benchmark
  Sharpe?" — the Probabilistic Sharpe Ratio with the benchmark as the
  hurdle — and requires the breach to **persist** for several consecutive
  evaluations before it becomes an alert. It also reports the standard
  error, so the reader can see how little a short window can say.

- **The benchmark must be the out-of-sample number.** Comparing live to an
  in-sample backtest Sharpe guarantees "decay" alerts, because live always
  looks worse than in-sample. Build the :class:`Benchmark` from the
  walk-forward / deflated result, not the headline backtest.

- **A drawdown 1.5x the backtest maximum is a post-mortem, not a tripwire.**
  Default here is 1.0x — you are at the worst the backtest ever saw, which
  is when to look, not after.

- **Zero activity is not decay.** A window with no trades has no Sharpe. The
  textbook version scores it 0 and halts; this raises a separate
  ``NO_ACTIVITY`` alert, because a strategy that stopped trading is a
  different problem from one that's losing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from backtest.deflated_sharpe import deannualize_sharpe, probabilistic_sharpe
from backtest.result import BacktestResult

Recommendation = Literal["insufficient_data", "continue", "review", "halt_recommended"]

_MIN_WINDOW = 2


@dataclass(frozen=True)
class Benchmark:
    """What live performance is compared against. Use OOS / deflated numbers.

    Attributes:
        sharpe_per_period: Expected per-period Sharpe. Convert annualized
            figures with :func:`deannualize_sharpe`.
        max_drawdown: Worst peak-to-trough decline, as a negative fraction.
        longest_underwater_bars: Longest run of bars below the running equity
            peak. ``None`` disables that check.
    """

    sharpe_per_period: float
    max_drawdown: float
    longest_underwater_bars: int | None = None

    def __post_init__(self) -> None:
        if self.max_drawdown > 0.0:
            msg = f"max_drawdown is a negative fraction, got {self.max_drawdown}"
            raise ValueError(msg)
        if self.longest_underwater_bars is not None and self.longest_underwater_bars < 0:
            msg = f"longest_underwater_bars must be >= 0, got {self.longest_underwater_bars}"
            raise ValueError(msg)

    @classmethod
    def from_result(cls, result: BacktestResult) -> Benchmark:
        """Build from a backtest — pass the walk-forward OOS result, not in-sample."""
        m = result.metrics
        return cls(
            sharpe_per_period=deannualize_sharpe(m.sharpe, result.config.periods_per_year),
            max_drawdown=m.max_drawdown,
            longest_underwater_bars=longest_underwater(result.ledger["equity"]),
        )


@dataclass(frozen=True)
class HealthConfig:
    """Evaluation parameters.

    Attributes:
        window: Trailing bars per evaluation.
        min_obs: Below this many live bars, report ``insufficient_data``.
        periods_per_year: For annualizing the reported live Sharpe.
        decay_alpha: PSR-vs-benchmark below this is a breach.
        consecutive_required: Breaches must persist this many consecutive
            evaluations (hysteresis against alert fatigue).
        drawdown_multiple: Current drawdown at or beyond this multiple of the
            benchmark's max is an alert. 1.0 = "at the worst ever seen".
        underwater_multiple: Bars underwater at or beyond this multiple of
            the benchmark's longest is an alert.
    """

    window: int = 30
    min_obs: int = 20
    periods_per_year: int = 365
    decay_alpha: float = 0.05
    consecutive_required: int = 3
    drawdown_multiple: float = 1.0
    underwater_multiple: float = 1.0

    def __post_init__(self) -> None:
        if self.window < _MIN_WINDOW:
            msg = f"window must be >= {_MIN_WINDOW}, got {self.window}"
            raise ValueError(msg)
        if not _MIN_WINDOW <= self.min_obs <= self.window:
            msg = f"min_obs must be in [{_MIN_WINDOW}, window], got {self.min_obs}"
            raise ValueError(msg)
        if self.periods_per_year <= 0:
            msg = f"periods_per_year must be > 0, got {self.periods_per_year}"
            raise ValueError(msg)
        if not 0.0 < self.decay_alpha < 1.0:
            msg = f"decay_alpha must be in (0, 1), got {self.decay_alpha}"
            raise ValueError(msg)
        if self.consecutive_required < 1:
            msg = f"consecutive_required must be >= 1, got {self.consecutive_required}"
            raise ValueError(msg)
        if self.drawdown_multiple <= 0.0 or self.underwater_multiple <= 0.0:
            msg = "drawdown_multiple and underwater_multiple must be > 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class HealthReport:
    """The monitor's output. A recommendation for a human, never an action.

    Attributes:
        n_obs: Live bars available.
        live_sharpe_annualized: Sharpe of the latest window, annualized.
        live_sharpe_se: Its standard error — read the Sharpe with this beside it.
        p_consistent: PSR of the latest window against the benchmark Sharpe:
            the probability the strategy still has its benchmark edge given
            this window. ``nan`` when the window had no activity.
        current_drawdown: Decline from the live running peak, negative fraction.
        bars_underwater: Consecutive bars below the live running peak.
        alerts: Zero or more of ``SHARPE_DECAY``, ``DRAWDOWN_EXCEEDED``,
            ``UNDERWATER_TOO_LONG``, ``NO_ACTIVITY``.
        recommendation: What the human should consider doing.
    """

    n_obs: int
    live_sharpe_annualized: float
    live_sharpe_se: float
    p_consistent: float
    current_drawdown: float
    bars_underwater: int
    alerts: tuple[str, ...]
    recommendation: Recommendation


def longest_underwater(equity: pd.Series) -> int:
    """Longest run of consecutive bars strictly below the running peak."""
    under = (equity < equity.cummax()).to_numpy()
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def _trailing_underwater(equity: pd.Series) -> int:
    """Bars since the equity was last at its running peak."""
    under = (equity < equity.cummax()).to_numpy()[::-1]
    run = 0
    for flag in under:
        if not flag:
            break
        run += 1
    return run


def _window_psr(window: pd.Series, benchmark_sr: float) -> float:
    """PSR of one window vs the benchmark; nan if the window had no activity."""
    std = float(window.std(ddof=1))
    if not std > 0.0:
        return math.nan
    sr = float(window.mean()) / std
    return probabilistic_sharpe(sr, len(window), benchmark=benchmark_sr)


def _decay_persists(r: pd.Series, benchmark_sr: float, cfg: HealthConfig) -> bool:
    """True if the trailing window breached at each of the last k end-points.

    Evaluates the window as it stood ending at the latest bar, the bar
    before, ... ``consecutive_required`` times; a breach only counts if every
    one of them breached. A single bad bar can't trip it.
    """
    n = len(r)
    for j in range(cfg.consecutive_required):
        end = n - j
        start = max(0, end - cfg.window)
        if end - start < cfg.min_obs:
            return False
        p = _window_psr(r.iloc[start:end], benchmark_sr)
        if math.isnan(p) or p >= cfg.decay_alpha:
            return False
    return True


def _recommend(alerts: list[str]) -> Recommendation:
    if "DRAWDOWN_EXCEEDED" in alerts:
        return "halt_recommended"
    return "review" if alerts else "continue"


def health_check(
    live_returns: pd.Series,
    benchmark: Benchmark,
    cfg: HealthConfig | None = None,
) -> HealthReport:
    """Score live returns against a benchmark and recommend — never act.

    Args:
        live_returns: Per-period net returns realized live, oldest first.
        benchmark: Out-of-sample expectations to compare against.
        cfg: Evaluation parameters.
    """
    if cfg is None:
        cfg = HealthConfig()
    r = live_returns.dropna()
    n = len(r)
    se = math.sqrt(cfg.periods_per_year / min(cfg.window, max(n, 1)))

    if n < cfg.min_obs:
        return HealthReport(
            n_obs=n,
            live_sharpe_annualized=math.nan,
            live_sharpe_se=se,
            p_consistent=math.nan,
            current_drawdown=0.0,
            bars_underwater=0,
            alerts=(),
            recommendation="insufficient_data",
        )

    alerts: list[str] = []

    # ── Sharpe decay, with persistence ──────────────────────────────────
    latest = r.iloc[max(0, n - cfg.window) :]
    p_latest = _window_psr(latest, benchmark.sharpe_per_period)
    if math.isnan(p_latest):
        alerts.append("NO_ACTIVITY")
    elif _decay_persists(r, benchmark.sharpe_per_period, cfg):
        alerts.append("SHARPE_DECAY")

    std_latest = float(latest.std(ddof=1))
    live_sr_ann = (
        float(latest.mean()) / std_latest * math.sqrt(cfg.periods_per_year)
        if std_latest > 0.0
        else math.nan
    )

    # ── Drawdown and time underwater ────────────────────────────────────
    equity = np.exp(r.cumsum())
    peak = equity.cummax()
    dd = float(equity.iloc[-1] / peak.iloc[-1] - 1.0)
    if benchmark.max_drawdown < 0.0 and dd <= cfg.drawdown_multiple * benchmark.max_drawdown:
        alerts.append("DRAWDOWN_EXCEEDED")

    underwater = _trailing_underwater(equity)
    if (
        benchmark.longest_underwater_bars is not None
        and benchmark.longest_underwater_bars > 0
        and underwater >= cfg.underwater_multiple * benchmark.longest_underwater_bars
    ):
        alerts.append("UNDERWATER_TOO_LONG")

    return HealthReport(
        n_obs=n,
        live_sharpe_annualized=live_sr_ann,
        live_sharpe_se=se,
        p_consistent=p_latest,
        current_drawdown=dd,
        bars_underwater=underwater,
        alerts=tuple(alerts),
        recommendation=_recommend(alerts),
    )
