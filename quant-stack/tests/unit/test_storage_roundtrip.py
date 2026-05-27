"""Round-trip tests for ParquetStore + PointInTimeQuery.

Writes records through ParquetStore, reads them back through a
PointInTimeQuery pinned to various decision times. Covers:

- Basic round trip per record type.
- Dedup within a batch and across batches.
- as_of pin filters out future rows.
- options chain refuses overwrite.
- naive datetime at query construction is rejected.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ingestion.schema import (
    CorporateAction,
    EarningsEvent,
    EquityBar,
    OptionContract,
    OptionRight,
    OptionsChainSnapshot,
)
from storage.query import PointInTimeQuery
from storage.store import ParquetStore


def _bar(
    *,
    symbol: str = "NVDA",
    as_of: datetime | None = None,
    interval: str = "1d",
    close: float = 100.0,
) -> EquityBar:
    return EquityBar(
        as_of=as_of or datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
        symbol=symbol,
        interval=interval,
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=1_000_000,
        source="schwab",
    )


# ─────────────────────────────────────────────────────────────────────────
#  Equity bars
# ─────────────────────────────────────────────────────────────────────────


def test_equity_bars_round_trip(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    bars = [
        _bar(as_of=datetime(2024, 1, 2, 21, 0, tzinfo=UTC), close=505),
        _bar(as_of=datetime(2024, 1, 3, 21, 0, tzinfo=UTC), close=510),
    ]
    result = store.write_equity_bars(bars)
    assert result.requested == 2
    assert result.persisted == 2
    assert result.deduplicated == 0
    assert len(result.files_touched) == 1

    q = PointInTimeQuery(store, as_of=datetime(2024, 1, 31, tzinfo=UTC))
    got = q.equity_bars("NVDA")
    assert len(got) == 2
    assert got[0].close == 505.0
    assert got[1].close == 510.0


def test_decision_time_filters_future(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    bars = [
        _bar(as_of=datetime(2024, 1, 2, 21, 0, tzinfo=UTC), close=100),
        _bar(as_of=datetime(2024, 1, 10, 21, 0, tzinfo=UTC), close=110),
        _bar(as_of=datetime(2024, 1, 20, 21, 0, tzinfo=UTC), close=120),
    ]
    store.write_equity_bars(bars)

    q = PointInTimeQuery(store, as_of=datetime(2024, 1, 11, tzinfo=UTC))
    got = q.equity_bars("NVDA")
    assert [b.close for b in got] == [100.0, 110.0]


def test_decision_time_before_all_data(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    store.write_equity_bars([_bar(as_of=datetime(2024, 6, 1, 21, 0, tzinfo=UTC))])
    q = PointInTimeQuery(store, as_of=datetime(2024, 1, 1, tzinfo=UTC))
    assert q.equity_bars("NVDA") == []


def test_dedup_within_batch(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    bar = _bar(as_of=datetime(2024, 1, 2, 21, 0, tzinfo=UTC))
    result = store.write_equity_bars([bar, bar, bar])
    assert result.requested == 3
    assert result.persisted == 1
    assert result.deduplicated == 2


def test_dedup_across_calls(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    bar = _bar(as_of=datetime(2024, 1, 2, 21, 0, tzinfo=UTC))
    r1 = store.write_equity_bars([bar])
    r2 = store.write_equity_bars([bar])
    assert r1.persisted == 1
    assert r2.persisted == 0
    assert r2.deduplicated == 1

    q = PointInTimeQuery(store, as_of=datetime(2024, 12, 31, tzinfo=UTC))
    assert len(q.equity_bars("NVDA")) == 1


def test_groups_by_year_and_interval(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    bars = [
        _bar(as_of=datetime(2023, 6, 1, 21, 0, tzinfo=UTC), interval="1d"),
        _bar(as_of=datetime(2024, 6, 1, 21, 0, tzinfo=UTC), interval="1d"),
        _bar(as_of=datetime(2024, 6, 1, 14, 30, tzinfo=UTC), interval="1m"),
    ]
    result = store.write_equity_bars(bars)
    assert result.persisted == 3
    # Three groups → three files
    assert len(result.files_touched) == 3


def test_latest_quote_respects_decision_time(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    bars = [
        _bar(as_of=datetime(2024, 1, 2, 21, 0, tzinfo=UTC), close=100),
        _bar(as_of=datetime(2024, 1, 10, 21, 0, tzinfo=UTC), close=110),
        _bar(as_of=datetime(2024, 1, 20, 21, 0, tzinfo=UTC), close=120),
    ]
    store.write_equity_bars(bars)
    q = PointInTimeQuery(store, as_of=datetime(2024, 1, 15, tzinfo=UTC))
    latest = q.latest_quote("NVDA")
    assert latest is not None
    assert latest.close == 110.0


def test_latest_quote_returns_none_with_no_data(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    q = PointInTimeQuery(store, as_of=datetime(2024, 1, 15, tzinfo=UTC))
    assert q.latest_quote("NVDA") is None


# ─────────────────────────────────────────────────────────────────────────
#  Options chain
# ─────────────────────────────────────────────────────────────────────────


def _contract(*, strike: float, right: OptionRight, expiry: date,
              as_of: datetime, ul: str = "NVDA") -> OptionContract:
    return OptionContract(
        as_of=as_of,
        underlying=ul,
        expiration=expiry,
        strike=strike,
        right=right,
        bid=1.0,
        ask=1.1,
        last=1.05,
        volume=10,
        open_interest=100,
        iv=0.30,
        delta=0.25,
        gamma=0.01,
        theta=-0.05,
        vega=0.10,
        rho=0.01,
        source="schwab",
    )


def test_options_chain_round_trip(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    snap_time = datetime(2024, 3, 15, 21, 0, tzinfo=UTC)
    snapshot = OptionsChainSnapshot(
        as_of=snap_time,
        underlying="NVDA",
        underlying_price=505.0,
        contracts=(
            _contract(
                strike=500, right=OptionRight.call,
                expiry=date(2024, 6, 21), as_of=snap_time,
            ),
            _contract(
                strike=510, right=OptionRight.put,
                expiry=date(2024, 6, 21), as_of=snap_time,
            ),
        ),
        source="schwab",
    )
    result = store.write_options_chain(snapshot)
    assert result.persisted == 2
    assert result.deduplicated == 0

    q = PointInTimeQuery(store, as_of=datetime(2024, 3, 16, tzinfo=UTC))
    got = q.options_chain_at("NVDA", date(2024, 3, 15))
    assert got is not None
    assert got.underlying == "NVDA"
    assert got.underlying_price == 505.0
    assert len(got.contracts) == 2


def test_options_chain_refuses_overwrite(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    snap_time = datetime(2024, 3, 15, 21, 0, tzinfo=UTC)
    snapshot = OptionsChainSnapshot(
        as_of=snap_time, underlying="NVDA", underlying_price=505.0,
        contracts=(
            _contract(
                strike=500, right=OptionRight.call,
                expiry=date(2024, 6, 21), as_of=snap_time,
            ),
        ),
        source="schwab",
    )
    store.write_options_chain(snapshot)
    # Second call: same path → silently dedups; never overwrites.
    second = store.write_options_chain(snapshot)
    assert second.persisted == 0
    assert second.deduplicated == 1


def test_options_chain_invisible_before_snapshot_time(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    snap_time = datetime(2024, 3, 15, 21, 0, tzinfo=UTC)
    snapshot = OptionsChainSnapshot(
        as_of=snap_time, underlying="NVDA", underlying_price=505.0,
        contracts=(
            _contract(
                strike=500, right=OptionRight.call,
                expiry=date(2024, 6, 21), as_of=snap_time,
            ),
        ),
        source="schwab",
    )
    store.write_options_chain(snapshot)
    # Decision time BEFORE the snapshot's as_of → invisible.
    q = PointInTimeQuery(store, as_of=datetime(2024, 3, 15, 14, 0, tzinfo=UTC))
    assert q.options_chain_at("NVDA", date(2024, 3, 15)) is None


# ─────────────────────────────────────────────────────────────────────────
#  Corporate actions + earnings
# ─────────────────────────────────────────────────────────────────────────


def test_corporate_actions_round_trip(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    actions = [
        CorporateAction(
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="NVDA",
            ex_date=date(2024, 6, 10),
            type="split",
            ratio=10.0,
            cash_amount=None,
            source="schwab",
        ),
    ]
    result = store.write_corporate_actions(actions)
    assert result.persisted == 1
    q = PointInTimeQuery(store, as_of=datetime(2024, 12, 31, tzinfo=UTC))
    got = q.corporate_actions("NVDA")
    assert len(got) == 1
    assert got[0].type == "split"
    assert got[0].ratio == 10.0


def test_earnings_round_trip(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    events = [
        EarningsEvent(
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="NVDA",
            report_date=date(2024, 2, 21),
            when="amc",
            eps_estimate=0.59,
            eps_actual=0.61,
            eps_surprise_pct=3.4,
            source="schwab",
        ),
    ]
    result = store.write_earnings(events)
    assert result.persisted == 1
    q = PointInTimeQuery(store, as_of=datetime(2024, 12, 31, tzinfo=UTC))
    got = q.earnings("NVDA")
    assert len(got) == 1
    assert got[0].eps_actual == 0.61


# ─────────────────────────────────────────────────────────────────────────
#  Defensive checks
# ─────────────────────────────────────────────────────────────────────────


def test_query_rejects_naive_decision_time(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    with pytest.raises(ValueError, match="timezone-aware"):
        PointInTimeQuery(store, as_of=datetime(2024, 1, 1))


def test_query_empty_when_symbol_missing(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    q = PointInTimeQuery(store, as_of=datetime(2024, 1, 1, tzinfo=UTC))
    assert q.equity_bars("DOES_NOT_EXIST") == []
    assert q.options_chain_at("DOES_NOT_EXIST", date(2024, 1, 1)) is None
    assert q.corporate_actions("DOES_NOT_EXIST") == []
    assert q.earnings("DOES_NOT_EXIST") == []


def test_store_requires_non_empty_env(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ParquetStore(tmp_path, env="")
