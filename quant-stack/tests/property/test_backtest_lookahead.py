"""Look-ahead property test for the vectorized backtester — Principle #1.

The backtester's central promise is that a position is taken on the bar AFTER
the signal is observed (``execution_lag`` bars later). Made concrete: perturbing
the signal at some future bar ``j`` must NEVER change any ledger row before bar
``j + execution_lag``. If it did, the past would depend on the future — the
canonical look-ahead bug.

We also check the dual with a hand-built case: a perturbation at bar ``j`` DOES
move the ledger from ``j + lag`` onward (otherwise the test could pass by the
engine simply ignoring the signal).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from backtest import BacktestConfig, run_backtest

_AFFECTED = ["position", "turnover", "gross", "funding", "costs", "net", "equity"]


@st.composite
def _scenario(draw: Any) -> tuple[pd.Series, pd.Series, int, float, int]:
    n = draw(st.integers(min_value=3, max_value=40))
    price_vals = draw(
        st.lists(
            st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    signal_vals = draw(
        st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    lag = draw(st.integers(min_value=1, max_value=3))
    perturb_at = draw(st.integers(min_value=0, max_value=n - 1))
    new_value = draw(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    )
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    prices = pd.Series(price_vals, index=idx)
    signal = pd.Series(signal_vals, index=idx)
    return prices, signal, perturb_at, new_value, lag


@given(scenario=_scenario())
@settings(max_examples=150, deadline=None)
def test_future_signal_never_moves_the_past(
    scenario: tuple[pd.Series, pd.Series, int, float, int],
) -> None:
    prices, signal, j, new_value, lag = scenario
    cfg = BacktestConfig(execution_lag=lag, funding_bps_per_year=100.0)

    base = run_backtest(prices, signal, cfg).ledger

    perturbed_signal = signal.copy()
    perturbed_signal.iloc[j] = new_value
    perturbed = run_backtest(prices, perturbed_signal, cfg).ledger

    # Rows strictly before j + lag must be byte-identical: the signal at bar j
    # cannot be acted on until bar j + lag.
    cutoff = j + lag
    if cutoff > 0:
        before = base[_AFFECTED].iloc[:cutoff]
        before_perturbed = perturbed[_AFFECTED].iloc[:cutoff]
        pd.testing.assert_frame_equal(before, before_perturbed)


def test_perturbation_does_take_effect_from_lag_onward() -> None:
    """Dual: the engine isn't passing the test by ignoring the signal."""
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    prices = pd.Series([100, 110, 121, 133.1, 146.41, 161.05], index=idx)
    signal = pd.Series(0.0, index=idx)
    cfg = BacktestConfig(execution_lag=1, fee_bps=0.0, slippage_bps=0.0)

    base = run_backtest(prices, signal, cfg).ledger

    perturbed_signal = signal.copy()
    perturbed_signal.iloc[2] = 1.0  # go long, observed at bar 2
    perturbed = run_backtest(prices, perturbed_signal, cfg).ledger

    # Position must differ at bar 3 (= 2 + lag), and be identical before.
    assert perturbed["position"].iloc[3] == 1.0
    assert base["position"].iloc[3] == 0.0
    assert np.array_equal(
        base["position"].iloc[:3].to_numpy(), perturbed["position"].iloc[:3].to_numpy()
    )
