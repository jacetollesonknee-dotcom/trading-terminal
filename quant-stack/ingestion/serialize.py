"""JSON-safe wire format for scheduler events.

The terminal (ADR-004) consumes the engine's :class:`~ingestion.scheduler.SchedulerEvent`
stream and pushes it to the browser over SocketIO. SocketIO needs plain
JSON, but a ``SchedulerEvent`` carries a :class:`datetime` and a
:class:`~storage.store.WriteResult` (which itself carries ``Path`` objects).

This module owns the one true mapping from those engine types to a
JSON-safe ``dict``. It lives in the engine — not the terminal — because
it is about the engine's own types, so it is gated and tested like the
rest of the engine. The terminal side is a thin, un-gated glue layer.

Design notes:
* Pure standard library. No Flask, no SocketIO — importable anywhere.
* The output schema is stable and documented below; the terminal's
  front-end depends on these key names.
* ``triggered_at`` is emitted as an ISO-8601 string in UTC.
"""

from __future__ import annotations

from typing import Any

from ingestion.scheduler import SchedulerEvent
from storage.store import WriteResult

__all__ = ["scheduler_event_to_dict", "write_result_to_dict"]


def write_result_to_dict(result: WriteResult) -> dict[str, Any]:
    """Convert a :class:`WriteResult` to a JSON-safe dict.

    ``files_touched`` is emitted as a count rather than a list of paths:
    the browser has no use for engine-side filesystem paths, and leaking
    them would be a mild information smell.
    """
    return {
        "requested": result.requested,
        "persisted": result.persisted,
        "deduplicated": result.deduplicated,
        "rejected_schema": result.rejected_schema,
        "files_touched": len(result.files_touched),
    }


def scheduler_event_to_dict(event: SchedulerEvent) -> dict[str, Any]:
    """Convert a :class:`SchedulerEvent` to a JSON-safe dict.

    Output schema (consumed by the terminal front-end)::

        {
          "source": "yahoo",           # which ingestion source
          "triggered_at": "2026-...Z", # ISO-8601 UTC
          "ok": true,                  # poll succeeded?
          "detail": "NVDA",            # ticker/handle/'latest' or null
          "error_message": null,       # populated when ok is false
          "write": {                   # null when the poll failed
            "requested": 1, "persisted": 1,
            "deduplicated": 0, "rejected_schema": 0,
            "files_touched": 1
          }
        }
    """
    return {
        "source": event.source,
        "triggered_at": event.triggered_at.isoformat(),
        "ok": event.ok,
        "detail": event.detail,
        "error_message": event.error_message,
        "write": (
            write_result_to_dict(event.write_result)
            if event.write_result is not None
            else None
        ),
    }
