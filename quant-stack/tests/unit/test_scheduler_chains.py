"""Chains tick tests: once per day, at/after the target time, gate-aware.

`run_capture` is patched at the tick's import point so nothing touches a
broker; `_utcnow` is patched so the clock is deterministic.
"""

from __future__ import annotations

import datetime as dt
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingestion.brokers.registry import BrokerDisabledError
from ingestion.chain_capture import CaptureResult
from ingestion.scheduler import IngestionScheduler, SchedulerConfig, SchedulerEvent
from storage.store import ParquetStore, WriteResult

_OK = WriteResult(requested=6, persisted=6, deduplicated=0, rejected_schema=0, files_touched=())


def _sched(
    tmp_path: Path,
    events: list[SchedulerEvent],
    *,
    watch: tuple[str, ...] = ("QQQ", "SPY"),
    after: dt.time = dt.time(20, 30),
) -> IngestionScheduler:
    cfg = SchedulerConfig(
        chains_enabled=True, chain_watchlist=watch, chains_capture_after_utc=after
    )
    store = ParquetStore(tmp_path, env="test")
    return IngestionScheduler(
        store, cfg, on_event=events.append, settings=object()  # type: ignore[arg-type]
    )


def _at(hour: int, minute: int = 0, day: int = 2) -> datetime:
    return datetime(2026, 3, day, hour, minute, tzinfo=UTC)


def _fake_capture(results: list[CaptureResult]):  # type: ignore[no-untyped-def]
    calls: list[tuple[str, ...]] = []

    def run_capture(_settings, _store, underlyings, *, broker_name="schwab", broker=None):  # type: ignore[no-untyped-def]
        calls.append(tuple(underlyings))
        return results

    return run_capture, calls


def test_not_due_before_target_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[SchedulerEvent] = []
    run, calls = _fake_capture([])
    monkeypatch.setattr("ingestion.chain_capture.run_capture", run)
    monkeypatch.setattr("ingestion.scheduler._utcnow", lambda: _at(15, 0))
    _sched(tmp_path, events)._tick_chains()
    assert calls == []
    assert events == []


def test_captures_once_per_day_after_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[SchedulerEvent] = []
    ok = [
        CaptureResult("QQQ", _at(20, 31), _OK),
        CaptureResult("SPY", _at(20, 31), _OK),
    ]
    run, calls = _fake_capture(ok)
    monkeypatch.setattr("ingestion.chain_capture.run_capture", run)
    monkeypatch.setattr("ingestion.scheduler._utcnow", lambda: _at(20, 31))
    s = _sched(tmp_path, events)

    s._tick_chains()
    s._tick_chains()  # same day: must not capture again
    assert calls == [("QQQ", "SPY")]
    assert [e.detail for e in events] == ["QQQ", "SPY"]
    assert all(e.ok and e.source == "chains" for e in events)
    assert s.chains_last_capture_date == _at(20, 31).date()

    # Next day, past the target: captures again.
    monkeypatch.setattr("ingestion.scheduler._utcnow", lambda: _at(20, 31, day=3))
    s._tick_chains()
    assert len(calls) == 2


def test_total_failure_leaves_day_unmarked_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[SchedulerEvent] = []
    bad = [CaptureResult("QQQ", _at(21, 0), None, error="RuntimeError: outage")]
    run, calls = _fake_capture(bad)
    monkeypatch.setattr("ingestion.chain_capture.run_capture", run)
    monkeypatch.setattr("ingestion.scheduler._utcnow", lambda: _at(21, 0))
    s = _sched(tmp_path, events, watch=("QQQ",))

    s._tick_chains()
    s._tick_chains()
    assert len(calls) == 2  # retried
    assert s.chains_last_capture_date is None
    assert events and not events[0].ok
    assert events[0].error_message == "RuntimeError: outage"


def test_partial_success_marks_day_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[SchedulerEvent] = []
    mixed = [
        CaptureResult("QQQ", _at(21, 0), _OK),
        CaptureResult("SPY", _at(21, 0), None, error="RuntimeError: nope"),
    ]
    run, calls = _fake_capture(mixed)
    monkeypatch.setattr("ingestion.chain_capture.run_capture", run)
    monkeypatch.setattr("ingestion.scheduler._utcnow", lambda: _at(21, 0))
    s = _sched(tmp_path, events)
    s._tick_chains()
    s._tick_chains()
    assert len(calls) == 1
    assert s.chains_last_capture_date == _at(21, 0).date()
    assert [e.ok for e in events] == [True, False]


def test_disabled_broker_emits_once_and_waits_for_tomorrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[SchedulerEvent] = []
    calls = 0

    def refuse(*_a, **_k):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        msg = "broker 'schwab' is not enabled."
        raise BrokerDisabledError(msg)

    monkeypatch.setattr("ingestion.chain_capture.run_capture", refuse)
    monkeypatch.setattr("ingestion.scheduler._utcnow", lambda: _at(21, 0))
    s = _sched(tmp_path, events)
    s._tick_chains()
    s._tick_chains()
    assert calls == 1
    assert len(events) == 1
    assert not events[0].ok
    assert "not enabled" in (events[0].error_message or "")
    assert events[0].detail == "QQQ,SPY"


def test_empty_chain_watchlist_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[SchedulerEvent] = []
    run, calls = _fake_capture([])
    monkeypatch.setattr("ingestion.chain_capture.run_capture", run)
    monkeypatch.setattr("ingestion.scheduler._utcnow", lambda: _at(21, 0))
    _sched(tmp_path, events, watch=())._tick_chains()
    assert calls == []


def test_chain_watchlist_is_dynamic_and_uppercased(tmp_path: Path) -> None:
    s = _sched(tmp_path, [], watch=())
    s.set_chain_watchlist(("qqq", "iren"))
    assert s.chain_watchlist() == ("QQQ", "IREN")


def test_chains_thread_spawned_only_when_enabled(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    off = IngestionScheduler(
        store,
        SchedulerConfig(yahoo_enabled=False, openinsider_enabled=False, zacks_enabled=False),
    )
    off.start()
    try:
        assert not off.running
    finally:
        off.stop(timeout=0.1)
