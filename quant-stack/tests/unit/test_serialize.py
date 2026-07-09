"""Unit tests for ingestion.serialize — the engine→terminal wire format."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ingestion.scheduler import SchedulerEvent
from ingestion.serialize import scheduler_event_to_dict, write_result_to_dict
from storage.store import WriteResult

_TS = datetime(2026, 7, 9, 14, 30, 0, tzinfo=UTC)


def test_write_result_to_dict_counts_files_not_paths() -> None:
    result = WriteResult(
        requested=3,
        persisted=2,
        deduplicated=1,
        rejected_schema=0,
        files_touched=(Path("a/posts.parquet"), Path("b/posts.parquet")),
    )
    out = write_result_to_dict(result)
    assert out == {
        "requested": 3,
        "persisted": 2,
        "deduplicated": 1,
        "rejected_schema": 0,
        "files_touched": 2,
    }
    # No raw Path leaks into the wire format.
    assert "a/posts.parquet" not in json.dumps(out)


def test_scheduler_event_ok_serializes_to_json() -> None:
    event = SchedulerEvent(
        source="yahoo",
        triggered_at=_TS,
        ok=True,
        write_result=WriteResult(requested=1, persisted=1, deduplicated=0, rejected_schema=0),
        detail="NVDA",
    )
    out = scheduler_event_to_dict(event)
    assert out["source"] == "yahoo"
    assert out["triggered_at"] == "2026-07-09T14:30:00+00:00"
    assert out["ok"] is True
    assert out["detail"] == "NVDA"
    assert out["error_message"] is None
    assert out["write"] == {
        "requested": 1,
        "persisted": 1,
        "deduplicated": 0,
        "rejected_schema": 0,
        "files_touched": 0,
    }
    # Must round-trip through JSON without a custom encoder.
    assert json.loads(json.dumps(out)) == out


def test_scheduler_event_failure_has_null_write() -> None:
    event = SchedulerEvent(
        source="zacks",
        triggered_at=_TS,
        ok=False,
        write_result=None,
        error_message="network timeout",
        detail="AAPL",
    )
    out = scheduler_event_to_dict(event)
    assert out["ok"] is False
    assert out["write"] is None
    assert out["error_message"] == "network timeout"
    assert json.loads(json.dumps(out)) == out
