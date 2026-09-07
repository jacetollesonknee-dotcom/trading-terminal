"""Walk-forward driver tests.

The two things that matter most: fit_fn never sees a test bar, and the
stitched OOS backtest carries positions across folds (no per-fold cold start).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import (
    BacktestConfig,
    FittedStrategy,
    WalkForwardConfig,
    walk_forward,
)


def _prices(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n))), index=idx)


def _always_long(_train: pd.Series) -> FittedStrategy:
    return FittedStrategy(signal_fn=lambda h: pd.Series(1.0, index=h.index), params={"k": 1.0})


def _no_cost() -> BacktestConfig:
    return BacktestConfig(fee_bps=0.0, slippage_bps=0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Fold geometry
# ─────────────────────────────────────────────────────────────────────────────


def test_fold_count_and_contiguity() -> None:
    prices = _prices(400)
    wf = WalkForwardConfig(train_bars=100, test_bars=50)
    res = walk_forward(prices, _always_long, _no_cost(), wf)
    # floor((400 - 100) / 50) = 6 folds; 400 = 100 + 6*50 exactly, no leftover.
    assert res.n_folds == 6
    for a, b in zip(res.folds, res.folds[1:], strict=False):
        # Test windows tile the OOS span with no gap and no overlap.
        assert prices.index.get_loc(b.test_start) == prices.index.get_loc(a.test_end) + 1
    assert res.folds[0].test_start == prices.index[100]
    assert res.folds[-1].test_end == prices.index[-1]


def test_leftover_bars_are_not_traded() -> None:
    prices = _prices(430)  # 30 trailing bars can't fill a 50-bar test window
    wf = WalkForwardConfig(train_bars=100, test_bars=50)
    res = walk_forward(prices, _always_long, _no_cost(), wf)
    assert res.n_folds == 6
    assert len(res.result.ledger) == 6 * 50


def test_rolling_train_window_has_fixed_length() -> None:
    seen: list[int] = []

    def spy(train: pd.Series) -> FittedStrategy:
        seen.append(len(train))
        return _always_long(train)

    walk_forward(_prices(400), spy, _no_cost(), WalkForwardConfig(100, 50, anchored=False))
    assert seen == [100] * 6


def test_anchored_train_window_expands() -> None:
    seen: list[int] = []

    def spy(train: pd.Series) -> FittedStrategy:
        seen.append(len(train))
        return _always_long(train)

    walk_forward(_prices(400), spy, _no_cost(), WalkForwardConfig(100, 50, anchored=True))
    assert seen == [100, 150, 200, 250, 300, 350]


# ─────────────────────────────────────────────────────────────────────────────
#  No leakage into fit; signal history is contiguous
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_never_sees_a_test_bar() -> None:
    prices = _prices(400)
    wf = WalkForwardConfig(train_bars=100, test_bars=50)
    fit_last_seen: list[pd.Timestamp] = []

    def spy(train: pd.Series) -> FittedStrategy:
        fit_last_seen.append(train.index[-1])
        return _always_long(train)

    res = walk_forward(prices, spy, _no_cost(), wf)
    for fold, last in zip(res.folds, fit_last_seen, strict=True):
        assert last == fold.train_end
        assert last < fold.test_start


def test_signal_history_runs_through_test_and_not_beyond() -> None:
    prices = _prices(400)
    wf = WalkForwardConfig(train_bars=100, test_bars=50)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fit(_train: pd.Series) -> FittedStrategy:
        def sig(h: pd.Series) -> pd.Series:
            spans.append((h.index[0], h.index[-1]))
            return pd.Series(1.0, index=h.index)

        return FittedStrategy(signal_fn=sig)

    res = walk_forward(prices, fit, _no_cost(), wf)
    for fold, (lo, hi) in zip(res.folds, spans, strict=True):
        assert lo == fold.train_start  # starts with the train window (warm-up)
        assert hi == fold.test_end  # ends exactly at the fold's last test bar


def test_lookback_indicator_is_warm_at_start_of_each_fold() -> None:
    """The cold-start fix: a 50-bar lookback on a 50-bar test window.

    Evaluated on the bare test slice, this signal would be NaN on 49 of 50
    bars per fold. With contiguous history it is defined on every test bar.
    """
    prices = _prices(400)
    wf = WalkForwardConfig(train_bars=100, test_bars=50)

    def fit(_train: pd.Series) -> FittedStrategy:
        def sig(h: pd.Series) -> pd.Series:
            ma = h.rolling(50).mean()
            return (h > ma).astype(float).where(ma.notna())

        return FittedStrategy(signal_fn=sig)

    res = walk_forward(prices, fit, _no_cost(), wf)
    # Position is the shifted signal; NaN signal -> 0. If the signal were
    # cold-started per fold, the first 49 bars of each fold would be forced
    # flat. Instead exposure is high across the whole OOS span.
    assert res.result.metrics.exposure > 0.3


# ─────────────────────────────────────────────────────────────────────────────
#  Stitched OOS backtest — positions carry across folds
# ─────────────────────────────────────────────────────────────────────────────


def test_positions_carry_across_fold_boundaries() -> None:
    """Always-long across 6 folds must trade ONCE, not once per fold."""
    prices = _prices(400)
    wf = WalkForwardConfig(train_bars=100, test_bars=50)
    res = walk_forward(prices, _always_long, BacktestConfig(), wf)
    # Only the initial entry on the first OOS bar; no phantom re-entries.
    assert res.result.metrics.num_trades == 1
    assert res.result.metrics.total_turnover == pytest.approx(1.0)


def test_stitched_result_equals_one_backtest_of_the_stitched_signal() -> None:
    """Sanity: the headline result IS a single run_backtest over the OOS span."""
    from backtest import run_backtest  # noqa: PLC0415

    prices = _prices(400)
    wf = WalkForwardConfig(train_bars=100, test_bars=50)
    cfg = BacktestConfig()
    res = walk_forward(prices, _always_long, cfg, wf)
    oos = prices.iloc[100:400]
    direct = run_backtest(oos, pd.Series(1.0, index=oos.index), cfg)
    pd.testing.assert_frame_equal(res.result.ledger, direct.ledger)


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def test_fold_metrics_and_param_table_shape() -> None:
    prices = _prices(400)
    wf = WalkForwardConfig(train_bars=100, test_bars=50)

    def fit(train: pd.Series) -> FittedStrategy:
        return FittedStrategy(
            signal_fn=lambda h: pd.Series(1.0, index=h.index),
            params={"mean": float(train.mean()), "std": float(train.std())},
        )

    res = walk_forward(prices, fit, _no_cost(), wf)
    assert len(res.fold_metrics) == 6
    assert {"sharpe", "total_return", "max_drawdown", "n_bars"} <= set(res.fold_metrics.columns)
    assert (res.fold_metrics["n_bars"] == 50).all()
    assert list(res.param_table.columns) == ["mean", "std"]
    assert len(res.param_table) == 6
    assert res.param_table.index.name == "fold"


def test_fold_metrics_are_rebased_per_fold() -> None:
    # Each fold's total_return must be its OWN return, not cumulative-from-start.
    prices = _prices(400)
    wf = WalkForwardConfig(train_bars=100, test_bars=50)
    res = walk_forward(prices, _always_long, _no_cost(), wf)
    net = res.result.ledger["net"]
    for fold in res.folds:
        expected = float(np.exp(net.loc[fold.test_start : fold.test_end].sum()) - 1.0)
        assert res.fold_metrics.loc[fold.number, "total_return"] == pytest.approx(expected)


def test_summary_keys_and_positive_folds() -> None:
    prices = _prices(400)
    res = walk_forward(prices, _always_long, _no_cost(), WalkForwardConfig(100, 50))
    s = res.summary()
    for k in ("oos_sharpe", "oos_cagr", "oos_max_drawdown", "n_folds", "positive_folds"):
        assert k in s
    assert 0 <= res.positive_folds <= res.n_folds
    assert s["oos_sharpe"] == res.result.metrics.sharpe


def test_flat_fold_counts_as_not_positive() -> None:
    # A strategy that's always flat has NaN fold Sharpes -> 0 positive folds.
    def flat(_train: pd.Series) -> FittedStrategy:
        return FittedStrategy(signal_fn=lambda h: pd.Series(0.0, index=h.index))

    res = walk_forward(_prices(400), flat, _no_cost(), WalkForwardConfig(100, 50))
    assert res.positive_folds == 0


def test_defaults_used_when_omitted() -> None:
    res = walk_forward(_prices(400), _always_long)
    assert res.result.config == BacktestConfig()
    assert res.n_folds == (400 - 180) // 60


# ─────────────────────────────────────────────────────────────────────────────
#  Validation — no silent fallbacks
# ─────────────────────────────────────────────────────────────────────────────


def test_rejects_too_few_bars_for_one_fold() -> None:
    with pytest.raises(ValueError, match="one fold"):
        walk_forward(_prices(100), _always_long, _no_cost(), WalkForwardConfig(100, 50))


def test_rejects_misaligned_signal() -> None:
    def bad(_train: pd.Series) -> FittedStrategy:
        # Drops the first bar -> different index than it was given.
        return FittedStrategy(signal_fn=lambda h: pd.Series(1.0, index=h.index[1:]))

    with pytest.raises(ValueError, match="same index"):
        walk_forward(_prices(400), bad, _no_cost(), WalkForwardConfig(100, 50))


def test_rejects_non_series_signal() -> None:
    def bad(_train: pd.Series) -> FittedStrategy:
        return FittedStrategy(signal_fn=lambda h: np.ones(len(h)))

    with pytest.raises(ValueError, match="same index"):
        walk_forward(_prices(400), bad, _no_cost(), WalkForwardConfig(100, 50))


@pytest.mark.parametrize(("train_bars", "test_bars"), [(1, 50), (100, 0), (0, 50)])
def test_rejects_bad_geometry(train_bars: int, test_bars: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        WalkForwardConfig(train_bars=train_bars, test_bars=test_bars)


def test_rejects_bad_prices() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    prices = pd.Series(100.0, index=idx)
    prices.iloc[10] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        walk_forward(prices, _always_long, _no_cost(), WalkForwardConfig(100, 50))
