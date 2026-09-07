"""Performance metrics computed from a backtest ledger.

Everything here is a pure function of the per-bar ledger plus the config's
annualization factor. Annualized figures use ``cfg.periods_per_year`` so that
Sharpe, volatility, and CAGR are all consistent with one another.

Degenerate cases (zero volatility, zero drawdown, no active bars) return
``nan`` rather than a fabricated number — a metric that isn't defined should
look undefined, not like a real result.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from backtest.config import BacktestConfig


@dataclass(frozen=True)
class PerformanceMetrics:
    """Summary statistics for one backtest run."""

    n_periods: int
    years: float
    final_equity: float
    total_return: float
    cagr: float
    ann_return: float
    ann_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    exposure: float
    total_turnover: float
    avg_turnover: float
    num_trades: int
    total_cost_drag: float

    def as_dict(self) -> dict[str, float | int]:
        """Metrics as a plain dict, handy for logging or serialization."""
        return asdict(self)


# math.exp overflows just above this exponent; annualized log-growth beyond it
# means an astronomically large return we report as +inf rather than crashing.
_EXP_OVERFLOW = 700.0


def _cagr(growth: float, years: float) -> float:
    """Compound annual growth rate, computed in log space to avoid overflow.

    ``growth`` is ``final_equity / initial_capital`` (always > 0 here since
    equity is ``initial * exp(...)``). Over very short windows with extreme
    returns the naive ``growth ** (1/years)`` overflows a float; the log-space
    form degrades to +inf / -1.0 gracefully instead.
    """
    if years <= 0 or growth <= 0:
        return math.nan
    log_cagr = math.log(growth) / years
    if log_cagr >= _EXP_OVERFLOW:
        return math.inf
    return math.expm1(log_cagr)


def _max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of the equity curve, as a negative fraction."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_metrics(ledger: pd.DataFrame, cfg: BacktestConfig) -> PerformanceMetrics:
    """Derive performance metrics from a ledger produced by ``run_backtest``.

    Args:
        ledger: The per-bar DataFrame (needs ``net``, ``position``, ``turnover``,
            ``costs``, ``equity`` columns).
        cfg: The config the run used — supplies the annualization factor and
            risk-free benchmark.
    """
    net = ledger["net"]
    equity = ledger["equity"]
    position = ledger["position"]
    turnover = ledger["turnover"]

    n_periods = len(ledger)
    ppy = cfg.periods_per_year
    years = n_periods / ppy

    final_equity = float(equity.iloc[-1])
    total_return = final_equity / cfg.initial_capital - 1.0
    cagr = _cagr(final_equity / cfg.initial_capital, years)

    # Log-return moments, annualized.
    mean_per_period = float(net.mean())
    std_per_period = float(net.std(ddof=1)) if n_periods > 1 else 0.0
    ann_return = mean_per_period * ppy
    ann_volatility = std_per_period * math.sqrt(ppy)

    excess = ann_return - cfg.risk_free_rate
    sharpe = excess / ann_volatility if ann_volatility > 0 else math.nan

    downside = net[net < 0]
    downside_std = float(downside.std(ddof=1)) * math.sqrt(ppy) if len(downside) > 1 else 0.0
    sortino = excess / downside_std if downside_std > 0 else math.nan

    max_dd = _max_drawdown(equity)
    calmar = cagr / abs(max_dd) if max_dd < 0 and not math.isnan(cagr) else math.nan

    active = position != 0
    n_active = int(active.sum())
    hit_rate = float((net[active] > 0).mean()) if n_active > 0 else math.nan
    exposure = n_active / n_periods if n_periods > 0 else 0.0

    total_turnover = float(turnover.sum())
    avg_turnover = float(turnover.mean())
    num_trades = int((turnover > 0).sum())
    total_cost_drag = float(ledger["costs"].sum() + ledger["funding"].sum())

    return PerformanceMetrics(
        n_periods=n_periods,
        years=years,
        final_equity=final_equity,
        total_return=total_return,
        cagr=cagr,
        ann_return=ann_return,
        ann_volatility=ann_volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        hit_rate=hit_rate,
        exposure=exposure,
        total_turnover=total_turnover,
        avg_turnover=avg_turnover,
        num_trades=num_trades,
        total_cost_drag=total_cost_drag,
    )
