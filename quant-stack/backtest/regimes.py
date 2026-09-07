"""Regime attribution: does the edge exist everywhere, or only when it's easy?

Labels every bar bull / bear / chop from its close relative to a trailing
moving average, then scores the backtest ledger within each regime. The
question this answers is not "what was the Sharpe" but "where did the
Sharpe come from" — an edge that only exists in one regime is a bet on that
regime persisting, and should be sized and described as such.

Labels are strictly trailing (the MA at bar *t* uses bars ``t-ma+1 .. t``),
so the attribution has no look-ahead. The first ``ma_bars - 1`` bars are
``unknown`` and excluded.

Per-regime bars are non-contiguous, so a drawdown is not defined for them
and is deliberately not reported. Sharpe and hit rate over the concatenated
per-regime returns are standard attribution practice; the annualization is
a convention for comparability, not a claim you could have traded that
regime in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from backtest.result import BacktestResult

_MIN_MA_BARS = 2


class Regime(StrEnum):
    bull = "bull"
    bear = "bear"
    chop = "chop"
    unknown = "unknown"


_LABELED = (Regime.bull, Regime.bear, Regime.chop)


@dataclass(frozen=True)
class RegimeConfig:
    """How bars are labeled.

    Attributes:
        ma_bars: Trailing moving-average length. 200 by convention.
        chop_band: Bars with ``|close / MA - 1| < chop_band`` are ``chop``;
            above the band is ``bull``, below is ``bear``. A dead band keeps a
            price straddling its MA from flipping regime every bar.
    """

    ma_bars: int = 200
    chop_band: float = 0.02

    def __post_init__(self) -> None:
        if self.ma_bars < _MIN_MA_BARS:
            msg = f"ma_bars must be >= {_MIN_MA_BARS}, got {self.ma_bars}"
            raise ValueError(msg)
        if self.chop_band < 0.0:
            msg = f"chop_band must be >= 0, got {self.chop_band}"
            raise ValueError(msg)


def classify_regimes(prices: pd.Series, cfg: RegimeConfig | None = None) -> pd.Series:
    """Label each bar of ``prices`` with a :class:`Regime` (as its string value)."""
    if cfg is None:
        cfg = RegimeConfig()
    ma = prices.rolling(cfg.ma_bars).mean()
    rel = prices / ma - 1.0
    labels = pd.Series(Regime.unknown.value, index=prices.index, dtype=object)
    known = ma.notna()
    labels[known & (rel >= cfg.chop_band)] = Regime.bull.value
    labels[known & (rel <= -cfg.chop_band)] = Regime.bear.value
    labels[known & (rel.abs() < cfg.chop_band)] = Regime.chop.value
    return labels


@dataclass(frozen=True)
class RegimeStats:
    """Attribution for one regime."""

    regime: Regime
    n_bars: int
    time_share: float
    sharpe: float
    total_return: float
    hit_rate: float
    exposure: float
    pnl_share: float


@dataclass(frozen=True)
class RegimeReport:
    """Per-regime attribution of a backtest ledger.

    Attributes:
        stats: One :class:`RegimeStats` per labeled regime, bull/bear/chop order.
        table: The same as a DataFrame indexed by regime.
        n_unknown: Bars excluded for MA warm-up.
    """

    stats: tuple[RegimeStats, ...]
    table: pd.DataFrame
    n_unknown: int

    @property
    def regimes_with_edge(self) -> tuple[Regime, ...]:
        """Regimes with a positive Sharpe and at least one bar."""
        return tuple(s.regime for s in self.stats if s.n_bars > 0 and s.sharpe > 0.0)

    @property
    def single_regime_edge(self) -> bool:
        """True if exactly one regime carries a positive Sharpe."""
        return len(self.regimes_with_edge) == 1

    def verdict(self) -> str:
        """One plain sentence about where the edge lives."""
        edge = self.regimes_with_edge
        if not edge:
            return "No regime shows a positive Sharpe. There is no edge to attribute."
        if len(edge) == 1:
            s = next(s for s in self.stats if s.regime is edge[0])
            return (
                f"Edge exists ONLY in {s.regime.value} ({s.time_share:.0%} of time, "
                f"{s.pnl_share:.0%} of P&L). This is a bet on that regime persisting."
            )
        return f"Edge is positive in {', '.join(r.value for r in edge)}."


def _stats_for(regime: Regime, net: pd.Series, position: pd.Series, n_total: int,
               total_pnl: float, ppy: int) -> RegimeStats:
    n = len(net)
    if n > 1 and float(net.std(ddof=1)) > 0.0:
        sharpe = float(net.mean()) / float(net.std(ddof=1)) * math.sqrt(ppy)
    else:
        sharpe = math.nan
    active = position != 0
    n_active = int(active.sum())
    return RegimeStats(
        regime=regime,
        n_bars=n,
        time_share=n / n_total if n_total > 0 else 0.0,
        sharpe=sharpe,
        total_return=float(math.expm1(float(net.sum()))) if n > 0 else 0.0,
        hit_rate=float((net[active] > 0).mean()) if n_active > 0 else math.nan,
        exposure=n_active / n if n > 0 else 0.0,
        pnl_share=float(net.sum()) / total_pnl if total_pnl != 0.0 else math.nan,
    )


def regime_report(
    result: BacktestResult,
    prices: pd.Series,
    cfg: RegimeConfig | None = None,
) -> RegimeReport:
    """Attribute a backtest's P&L to bull / bear / chop regimes.

    Args:
        result: The backtest to attribute.
        prices: A price series whose index CONTAINS the ledger's index. It
            may (and for a walk-forward ledger should) extend earlier, so the
            moving average is already warm when the ledger begins.
        cfg: Labeling rules. Defaults to a 200-bar MA with a 2% dead band.
    """
    if cfg is None:
        cfg = RegimeConfig()
    ledger = result.ledger
    if not ledger.index.isin(prices.index).all():
        msg = "prices must contain every timestamp in the ledger"
        raise ValueError(msg)

    labels = classify_regimes(prices, cfg).loc[ledger.index]
    known = labels != Regime.unknown.value
    net_all = ledger["net"][known]
    n_total = int(known.sum())
    total_pnl = float(net_all.sum())
    ppy = result.config.periods_per_year

    stats = tuple(
        _stats_for(
            regime,
            ledger["net"][labels == regime.value],
            ledger["position"][labels == regime.value],
            n_total,
            total_pnl,
            ppy,
        )
        for regime in _LABELED
    )
    table = pd.DataFrame(
        [
            {
                "regime": s.regime.value,
                "n_bars": s.n_bars,
                "time_share": s.time_share,
                "sharpe": s.sharpe,
                "total_return": s.total_return,
                "hit_rate": s.hit_rate,
                "exposure": s.exposure,
                "pnl_share": s.pnl_share,
            }
            for s in stats
        ]
    ).set_index("regime")
    return RegimeReport(stats=stats, table=table, n_unknown=int((~known).sum()))
