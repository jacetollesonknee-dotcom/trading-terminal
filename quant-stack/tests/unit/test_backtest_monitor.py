"""Health-monitor tests.

The headline: a healthy strategy must NOT be flagged most of the time, and
the monitor must never do anything but recommend.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest import (
    BacktestConfig,
    Benchmark,
    HealthConfig,
    HealthReport,
    deannualize_sharpe,
    health_check,
    longest_underwater,
    run_backtest,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")


def _bench(sharpe_ann: float = 1.5, max_dd: float = -0.20) -> Benchmark:
    return Benchmark(
        sharpe_per_period=deannualize_sharpe(sharpe_ann, 365),
        max_drawdown=max_dd,
        longest_underwater_bars=60,
    )


def _live(mean: float, std: float, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n), index=_idx(n))


# ─────────────────────────────────────────────────────────────────────────────
#  False alarms: the whole point
# ─────────────────────────────────────────────────────────────────────────────


def test_healthy_strategy_rarely_flagged_for_decay() -> None:
    """True Sharpe 1.5, 200 independent 90-bar histories. The textbook rule
    fires ~40% of the time; this must stay in single digits."""
    bench = _bench(1.5)
    sd = 0.02
    mu = bench.sharpe_per_period * sd
    fired = 0
    trials = 200
    for seed in range(trials):
        rep = health_check(_live(mu, sd, 90, seed), bench, HealthConfig(window=30))
        fired += "SHARPE_DECAY" in rep.alerts
    assert fired / trials < 0.10


def test_dead_strategy_is_eventually_flagged() -> None:
    # Benchmark says Sharpe 3 (per-period 0.157); live is strongly negative.
    # A very deep DD benchmark isolates the decay check from the DD check.
    bench = _bench(3.0, max_dd=-0.95)
    rep = health_check(_live(-0.006, 0.02, 120, 1), bench, HealthConfig())
    assert "SHARPE_DECAY" in rep.alerts
    assert "DRAWDOWN_EXCEEDED" not in rep.alerts
    assert rep.recommendation == "review"


def test_breach_must_persist() -> None:
    """A single catastrophic bar breaches the latest window only.

    The windows ending one and two bars earlier don't contain it and look
    healthy, so with consecutive_required=3 there is no alert — the breach
    hasn't persisted. With consecutive_required=1 it fires immediately.
    """
    bench = _bench(3.0, max_dd=-0.95)  # deep DD benchmark isolates the decay check
    series = _live(0.004, 0.02, 70, 2)
    series.iloc[-1] = -0.5  # one-bar disaster at the very end
    one = health_check(series, bench, HealthConfig(consecutive_required=1, decay_alpha=0.10))
    three = health_check(series, bench, HealthConfig(consecutive_required=3, decay_alpha=0.10))
    assert "SHARPE_DECAY" in one.alerts
    assert "SHARPE_DECAY" not in three.alerts
    assert isinstance(three, HealthReport)


# ─────────────────────────────────────────────────────────────────────────────
#  Other alerts
# ─────────────────────────────────────────────────────────────────────────────


def test_no_activity_is_its_own_alert_not_decay() -> None:
    live = pd.Series(0.0, index=_idx(40))
    rep = health_check(live, _bench(), HealthConfig())
    assert "NO_ACTIVITY" in rep.alerts
    assert "SHARPE_DECAY" not in rep.alerts
    assert math.isnan(rep.p_consistent)
    assert rep.recommendation == "review"


def test_drawdown_breach_recommends_halt() -> None:
    # Ramp up then crash 30% while the benchmark max was -20%.
    up = [0.005] * 40
    crash = [-0.03] * 12  # ~ -30%
    live = pd.Series(up + crash, index=_idx(52))
    rep = health_check(live, _bench(max_dd=-0.20), HealthConfig())
    assert "DRAWDOWN_EXCEEDED" in rep.alerts
    assert rep.recommendation == "halt_recommended"
    assert rep.current_drawdown < -0.20


def test_drawdown_multiple_is_respected() -> None:
    up = [0.005] * 40
    dip = [-0.02] * 12  # ~ -21%
    live = pd.Series(up + dip, index=_idx(52))
    at_one = health_check(live, _bench(max_dd=-0.20), HealthConfig(drawdown_multiple=1.0))
    at_two = health_check(live, _bench(max_dd=-0.20), HealthConfig(drawdown_multiple=2.0))
    assert "DRAWDOWN_EXCEEDED" in at_one.alerts
    assert "DRAWDOWN_EXCEEDED" not in at_two.alerts


def test_underwater_too_long() -> None:
    # Peak, then 70 bars of tiny losses (never a new peak) vs benchmark longest 60.
    live = pd.Series([0.01] * 5 + [-0.0001] * 70, index=_idx(75))
    rep = health_check(live, _bench(), HealthConfig(min_obs=20))
    assert "UNDERWATER_TOO_LONG" in rep.alerts
    assert rep.bars_underwater == 70


def test_insufficient_data_makes_no_claims() -> None:
    rep = health_check(_live(0.0, 0.02, 5, 0), _bench(), HealthConfig())
    assert rep.recommendation == "insufficient_data"
    assert rep.alerts == ()
    assert math.isnan(rep.live_sharpe_annualized)


def test_reports_standard_error_of_live_sharpe() -> None:
    rep = health_check(_live(0.001, 0.02, 90, 0), _bench(), HealthConfig(window=30))
    assert rep.live_sharpe_se == pytest.approx(math.sqrt(252 / 30))


def test_recommendation_vocabulary_never_includes_an_action() -> None:
    # The monitor's whole contract: it recommends, a human acts.
    rep = health_check(_live(0.001, 0.02, 90, 0), _bench(), HealthConfig())
    assert rep.recommendation in {"insufficient_data", "continue", "review", "halt_recommended"}


# ─────────────────────────────────────────────────────────────────────────────
#  Benchmark construction
# ─────────────────────────────────────────────────────────────────────────────


def test_longest_underwater() -> None:
    eq = pd.Series([1.0, 1.1, 1.0, 0.9, 1.05, 1.2, 1.1], index=_idx(7))
    # Below running peak at bars 2,3,4 (3 bars), then 6 (1 bar).
    assert longest_underwater(eq) == 3


def test_benchmark_from_result() -> None:
    rng = np.random.default_rng(3)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 400))), index=_idx(400))
    result = run_backtest(prices, pd.Series(1.0, index=prices.index), BacktestConfig())
    b = Benchmark.from_result(result)
    assert b.sharpe_per_period == pytest.approx(result.metrics.sharpe / math.sqrt(252))
    assert b.max_drawdown == result.metrics.max_drawdown
    assert b.longest_underwater_bars is not None and b.longest_underwater_bars >= 0


def test_benchmark_rejects_positive_drawdown() -> None:
    with pytest.raises(ValueError, match="negative fraction"):
        Benchmark(sharpe_per_period=0.05, max_drawdown=0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 1},
        {"min_obs": 1},
        {"min_obs": 50},  # > window
        {"periods_per_year": 0},
        {"decay_alpha": 0.0},
        {"decay_alpha": 1.0},
        {"consecutive_required": 0},
        {"drawdown_multiple": 0.0},
    ],
)
def test_rejects_bad_health_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        HealthConfig(**kwargs)  # type: ignore[arg-type]
