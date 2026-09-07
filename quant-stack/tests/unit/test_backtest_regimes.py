"""Regime attribution tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest import (
    BacktestConfig,
    Regime,
    RegimeConfig,
    classify_regimes,
    regime_report,
    run_backtest,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")


def test_labels_follow_close_vs_ma_with_dead_band() -> None:
    # Flat at 100 for the MA warm-up, then jump up, then jump down, then back near.
    vals = [100.0] * 10 + [110.0] * 3 + [80.0] * 3 + [100.5] * 3
    prices = pd.Series(vals, index=_idx(len(vals)))
    labels = classify_regimes(prices, RegimeConfig(ma_bars=10, chop_band=0.02))
    assert (labels.iloc[:9] == Regime.unknown.value).all()  # MA warm-up
    assert labels.iloc[9] == Regime.chop.value  # 100 vs MA 100
    assert labels.iloc[10] == Regime.bull.value  # 110 vs MA ~101
    assert labels.iloc[13] == Regime.bear.value  # 80 vs MA ~103


def test_labels_are_trailing_no_lookahead() -> None:
    # Changing a FUTURE price must not change any earlier label.
    rng = np.random.default_rng(0)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))), index=_idx(300))
    base = classify_regimes(prices, RegimeConfig(ma_bars=50))
    bumped = prices.copy()
    bumped.iloc[200] *= 5.0
    after = classify_regimes(bumped, RegimeConfig(ma_bars=50))
    assert (base.iloc[:200] == after.iloc[:200]).all()


def test_report_attributes_pnl_and_time_shares_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 600))), index=_idx(600))
    signal = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signal, BacktestConfig(fee_bps=0, slippage_bps=0))
    rep = regime_report(result, prices, RegimeConfig(ma_bars=200))
    assert rep.n_unknown == 199
    assert rep.table["time_share"].sum() == pytest.approx(1.0)
    assert rep.table["n_bars"].sum() == 600 - 199
    # P&L shares sum to 1 when total P&L is non-zero.
    if not math.isnan(rep.table["pnl_share"].iloc[0]):
        assert rep.table["pnl_share"].sum() == pytest.approx(1.0)


def test_single_regime_edge_is_detected() -> None:
    """A long-only strategy on a series that only rises above its MA in one stretch."""
    # Flat 250 bars (chop), then a JUMP 5% above the MA and a trend (bull),
    # then flat again. The jump matters: a gradual ramp would spend its
    # first positive-return bars inside the dead band, labeled chop.
    up = list(105 * np.exp(np.linspace(0, 0.5, 100)))
    vals = [100.0] * 250 + up + [up[-1]] * 250
    prices = pd.Series(vals, index=_idx(len(vals)))
    signal = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signal, BacktestConfig(fee_bps=0, slippage_bps=0))
    rep = regime_report(result, prices, RegimeConfig(ma_bars=200, chop_band=0.02))
    # All the P&L happened while price was above its MA -> bull only.
    assert Regime.bull in rep.regimes_with_edge
    assert rep.single_regime_edge
    assert "ONLY in bull" in rep.verdict()


def test_single_regime_edge_with_net_loss_reads_sensibly() -> None:
    """Profitable in bull, but a bigger loss elsewhere: no '-136% of P&L'."""
    # Flat, then jump+trend up (bull, profitable), then a crash below the MA
    # (bear) that erases more than the bull gained. Long-only rides both.
    up = list(105 * np.exp(np.linspace(0, 0.3, 60)))
    down = list(up[-1] * np.exp(np.linspace(0, -0.8, 60)))
    vals = [100.0] * 250 + up + down + [down[-1]] * 200
    prices = pd.Series(vals, index=_idx(len(vals)))
    signal = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signal, BacktestConfig(fee_bps=0, slippage_bps=0))
    rep = regime_report(result, prices, RegimeConfig(ma_bars=200, chop_band=0.02))
    assert result.metrics.total_return < 0
    assert rep.single_regime_edge
    v = rep.verdict()
    assert "ONLY in bull" in v
    assert "lost money overall" in v
    assert "%" not in v.split("of time")[1]  # no negative-share nonsense after the time share


def test_no_edge_verdict() -> None:
    prices = pd.Series(100.0, index=_idx(300))
    prices = prices * (1 + 0.001 * np.sin(np.arange(300)))  # tiny wiggle around MA
    signal = pd.Series(0.0, index=prices.index)
    result = run_backtest(prices, signal)
    rep = regime_report(result, prices, RegimeConfig(ma_bars=50))
    assert rep.regimes_with_edge == ()
    assert "no edge" in rep.verdict().lower()


def test_prices_may_extend_before_ledger() -> None:
    # Walk-forward ledgers start after a train window; the MA should use it.
    rng = np.random.default_rng(2)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))), index=_idx(500))
    oos = prices.iloc[300:]
    result = run_backtest(oos, pd.Series(1.0, index=oos.index))
    rep = regime_report(result, prices, RegimeConfig(ma_bars=200))
    assert rep.n_unknown == 0  # MA already warm from the earlier bars


def test_rejects_prices_missing_ledger_bars() -> None:
    prices = pd.Series(100.0, index=_idx(300))
    result = run_backtest(prices, pd.Series(1.0, index=prices.index))
    with pytest.raises(ValueError, match="must contain"):
        regime_report(result, prices.iloc[:100])


@pytest.mark.parametrize("kwargs", [{"ma_bars": 1}, {"chop_band": -0.1}])
def test_rejects_bad_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        RegimeConfig(**kwargs)  # type: ignore[arg-type]
