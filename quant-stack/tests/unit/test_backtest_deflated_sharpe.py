"""Deflated Sharpe Ratio tests.

Beyond basic correctness, these pin down the two traps the naive one-liner
falls into: annualized-vs-per-period units, and the missing sqrt(V) scale on
the expected-max term.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestConfig, run_backtest
from backtest.deflated_sharpe import (
    deannualize_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)


def _returns(mean: float, std: float, n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(rng.normal(mean, std, n), index=idx)


# ─────────────────────────────────────────────────────────────────────────────
#  expected_max_sharpe (SR*)
# ─────────────────────────────────────────────────────────────────────────────


def test_single_trial_has_no_selection_effect() -> None:
    # The raw formula gives -inf here (Z(1 - 1/1) = Z(0)); we return 0.
    assert expected_max_sharpe(1, 0.5) == 0.0


def test_zero_dispersion_has_no_selection_effect() -> None:
    assert expected_max_sharpe(100, 0.0) == 0.0


def test_expected_max_scales_with_trial_dispersion() -> None:
    # SR* is sqrt(V) * E[max]; doubling the std must double SR*.
    assert expected_max_sharpe(20, 0.2) == pytest.approx(2 * expected_max_sharpe(20, 0.1))


def test_expected_max_grows_with_trials() -> None:
    assert expected_max_sharpe(100, 0.1) > expected_max_sharpe(10, 0.1) > 0.0


def test_expected_max_matches_paper_bracket() -> None:
    # With unit std the result is the paper's E[max of N std normals] bracket.
    n = 20
    g = 0.5772156649015329
    from scipy.stats import norm  # noqa: PLC0415  (test-local)

    bracket = (1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e))
    assert expected_max_sharpe(n, 1.0) == pytest.approx(float(bracket))


@pytest.mark.parametrize("bad", [0, -1])
def test_expected_max_rejects_bad_trials(bad: int) -> None:
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(bad, 0.1)


# ─────────────────────────────────────────────────────────────────────────────
#  probabilistic_sharpe (PSR)
# ─────────────────────────────────────────────────────────────────────────────


def test_psr_of_zero_sharpe_is_one_half() -> None:
    assert probabilistic_sharpe(0.0, 1000) == pytest.approx(0.5)


def test_psr_rises_with_more_observations() -> None:
    assert probabilistic_sharpe(0.05, 2000) > probabilistic_sharpe(0.05, 200)


def test_psr_falls_with_higher_benchmark() -> None:
    assert probabilistic_sharpe(0.1, 500, benchmark=0.0) > probabilistic_sharpe(
        0.1, 500, benchmark=0.05
    )


def test_psr_rejects_pathological_moments() -> None:
    # Large positive skew * sharpe drives the variance term negative.
    with pytest.raises(ValueError, match="non-positive"):
        probabilistic_sharpe(2.0, 100, skew=5.0, kurtosis=3.0)


def test_psr_rejects_too_few_obs() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        probabilistic_sharpe(0.1, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  The units trap
# ─────────────────────────────────────────────────────────────────────────────


def test_annualized_sharpe_overstates_significance() -> None:
    """Feeding an annualized Sharpe into a per-period test inflates it.

    This is the bug the naive one-liner has: same strategy, same data, but the
    p-value comes out wildly different depending on the units you (wrongly)
    pass in. The correct answer is the per-period one.
    """
    ppy = 365
    sharpe_ann = 1.5
    n_obs = 730
    per_period = deannualize_sharpe(sharpe_ann, ppy)
    assert per_period == pytest.approx(1.5 / math.sqrt(365))
    wrong = probabilistic_sharpe(sharpe_ann, n_obs)
    right = probabilistic_sharpe(per_period, n_obs)
    assert wrong > 0.9999  # "certainly significant"
    assert right < wrong  # and materially less confident once units are right
    assert 0.5 < right < 1.0


def test_deannualize_rejects_bad_periods() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        deannualize_sharpe(1.0, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  deflated_sharpe end to end
# ─────────────────────────────────────────────────────────────────────────────


def test_deflation_only_ever_reduces_confidence() -> None:
    r = _returns(0.001, 0.02, 730)
    alone = deflated_sharpe(r, trial_sharpes=[0.05])
    # Add 49 more trials with real dispersion around the same level.
    trials = [0.05, *np.random.default_rng(1).normal(0.0, 0.05, 49).tolist()]
    deflated = deflated_sharpe(r, trial_sharpes=trials)
    assert deflated.expected_max_sharpe > 0.0
    assert deflated.deflated_sharpe < alone.deflated_sharpe


def test_single_trial_reduces_to_psr() -> None:
    r = _returns(0.001, 0.02, 500)
    res = deflated_sharpe(r, trial_sharpes=[0.05])
    assert res.expected_max_sharpe == 0.0
    assert res.n_trials == 1
    assert res.deflated_sharpe == pytest.approx(
        probabilistic_sharpe(res.sharpe_per_period, res.n_obs, skew=res.skew, kurtosis=res.kurtosis)
    )


def test_pure_noise_does_not_pass() -> None:
    # Zero-mean returns picked as "best of 200" must be rejected.
    r = _returns(0.0, 0.02, 730, seed=3)
    trials = np.random.default_rng(4).normal(0.0, 0.04, 200).tolist()
    res = deflated_sharpe(r, trial_sharpes=trials)
    assert not res.passes
    assert res.deflated_sharpe < 0.95


def test_uses_raw_kurtosis_convention() -> None:
    # Normal returns have raw kurtosis ~3 (excess ~0). If we'd wrongly pulled
    # scipy's excess default, this would read ~0.
    r = _returns(0.0, 0.02, 20_000, seed=5)
    res = deflated_sharpe(r, trial_sharpes=[0.0])
    assert res.kurtosis == pytest.approx(3.0, abs=0.15)


def test_integrates_with_backtest_ledger() -> None:
    idx = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(6)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 400))), index=idx)
    signal = pd.Series(1.0, index=idx)
    result = run_backtest(prices, signal, BacktestConfig())
    # Bridge from the framework's annualized Sharpe to per-period trial units.
    trial = deannualize_sharpe(result.metrics.sharpe, result.config.periods_per_year)
    res = deflated_sharpe(result.ledger["net"], trial_sharpes=[trial])
    assert 0.0 <= res.deflated_sharpe <= 1.0
    assert res.n_obs == 400


def test_passes_property_respects_threshold() -> None:
    r = _returns(0.003, 0.01, 2000, seed=7)  # strong, clean edge
    res = deflated_sharpe(r, trial_sharpes=[0.3], threshold=0.5)
    assert res.passes


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5, -0.1])
def test_rejects_bad_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        deflated_sharpe(_returns(0.0, 0.01, 50), [0.0], threshold=threshold)


def test_rejects_empty_trials() -> None:
    with pytest.raises(ValueError, match="at least"):
        deflated_sharpe(_returns(0.0, 0.01, 50), [])


def test_rejects_nan_trials() -> None:
    with pytest.raises(ValueError, match="NaN"):
        deflated_sharpe(_returns(0.0, 0.01, 50), [0.1, float("nan")])


def test_rejects_too_few_returns() -> None:
    with pytest.raises(ValueError, match="observations"):
        deflated_sharpe(_returns(0.0, 0.01, 3), [0.0])


def test_rejects_constant_returns() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    with pytest.raises(ValueError, match="zero variance"):
        deflated_sharpe(pd.Series(0.01, index=idx), [0.0])
