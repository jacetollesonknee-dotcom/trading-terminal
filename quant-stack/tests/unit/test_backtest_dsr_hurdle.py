"""minimum_sharpe_to_pass tests."""

from __future__ import annotations

import math

import pytest

from backtest.deflated_sharpe import (
    expected_max_sharpe,
    minimum_sharpe_to_pass,
    probabilistic_sharpe,
)


def test_returned_sharpe_clears_threshold_exactly() -> None:
    n_obs, n_trials = 1000, 20
    sr = minimum_sharpe_to_pass(n_obs, n_trials)
    sr_star = expected_max_sharpe(n_trials, 1 / math.sqrt(n_obs))
    assert probabilistic_sharpe(sr, n_obs, benchmark=sr_star) == pytest.approx(0.95, abs=1e-6)
    assert probabilistic_sharpe(sr * 0.99, n_obs, benchmark=sr_star) < 0.95


def test_hurdle_rises_with_trials() -> None:
    n = 1000
    one, ten, hundred = (minimum_sharpe_to_pass(n, k) for k in (1, 10, 100))
    assert one < ten < hundred


def test_hurdle_falls_with_observations() -> None:
    assert minimum_sharpe_to_pass(4000, 10) < minimum_sharpe_to_pass(400, 10)


def test_single_trial_matches_plain_psr_hurdle() -> None:
    # N=1 -> SR*=0 -> the hurdle is just z(0.95)/sqrt(n-1) to first order.
    n = 2000
    sr = minimum_sharpe_to_pass(n, 1)
    assert sr == pytest.approx(1.6449 / math.sqrt(n - 1), rel=0.02)


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        minimum_sharpe_to_pass(1, 5)
    with pytest.raises(ValueError, match="threshold"):
        minimum_sharpe_to_pass(100, 5, threshold=1.0)
