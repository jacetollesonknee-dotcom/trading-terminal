"""Episodic SQLite migration + DDL tests.

We apply the real migration into an in-memory SQLite, then exercise:
- naked-only CHECK constraint on trades.structure
- proposed_updates.status enum constraint
- session/ts composite index on conversations
- idempotency of apply_migrations
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memory.episodic import (
    MigrationError,
    apply_migrations,
    get_connection,
    initialize,
)


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = initialize(":memory:")
    yield c
    c.close()


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ─────────────────────────────────────────────────────────────────────────
#  Migration mechanics
# ─────────────────────────────────────────────────────────────────────────

def test_initial_migration_runs(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT version, filename FROM schema_versions").fetchall()
    assert len(rows) == 1
    assert rows[0]["version"] == 1
    assert rows[0]["filename"] == "001_init.sql"


def test_apply_migrations_is_idempotent(conn: sqlite3.Connection) -> None:
    applied_second_time = apply_migrations(conn)
    assert applied_second_time == []  # nothing new to do


def test_all_six_tables_exist(conn: sqlite3.Connection) -> None:
    expected = {"trades", "conversations", "positions", "signals_fired",
                "mistakes", "proposed_updates", "schema_versions"}
    actual = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert expected <= actual


# ─────────────────────────────────────────────────────────────────────────
#  Naked-only enforcement at the DB layer
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("structure",
                         ["long_call", "long_put", "cash_secured_put", "covered_call"])
def test_trades_accepts_naked_structures(conn: sqlite3.Connection, structure: str) -> None:
    conn.execute(
        "INSERT INTO trades(ts, ticker, structure) VALUES (?, ?, ?)",
        (_now(), "NVDA", structure),
    )


@pytest.mark.parametrize("structure",
                         ["vertical", "iron_condor", "naked_short_call",
                          "short_strangle", "calendar"])
def test_trades_rejects_disallowed_structures(
    conn: sqlite3.Connection, structure: str
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO trades(ts, ticker, structure) VALUES (?, ?, ?)",
            (_now(), "NVDA", structure),
        )


# ─────────────────────────────────────────────────────────────────────────
#  proposed_updates status enum
# ─────────────────────────────────────────────────────────────────────────

def test_proposed_updates_status_default(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO proposed_updates(ts, file, diff) VALUES (?, ?, ?)",
        (_now(), "strategies.yaml", "--- a/x\n+++ b/x\n"),
    )
    row = conn.execute("SELECT status FROM proposed_updates").fetchone()
    assert row["status"] == "pending"


def test_proposed_updates_status_constrained(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO proposed_updates(ts, file, diff, status) VALUES (?, ?, ?, ?)",
            (_now(), "strategies.yaml", "...", "yolo"),
        )


# ─────────────────────────────────────────────────────────────────────────
#  conversations / signals_fired / mistakes basic write
# ─────────────────────────────────────────────────────────────────────────

def test_conversations_round_trip(conn: sqlite3.Connection) -> None:
    cid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO conversations(id, ts, session_id, role, content) "
        "VALUES (?, ?, ?, ?, ?)",
        (cid, _now(), "sess-1", "user", "what's my NVDA exposure?"),
    )
    row = conn.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
    assert row["role"] == "user"


def test_signals_fired_action_taken_constrained(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO signals_fired(ts, signal_name, score, action_taken, inputs_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (_now(), "ai_infra_premium_harvest_csp", 0.72, "proposed",
         json.dumps({"iv_rank": 65, "regime": "low_vol"})),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO signals_fired(ts, signal_name, score, action_taken) "
            "VALUES (?, ?, ?, ?)",
            (_now(), "x", 0.1, "executed_anyway"),
        )


def test_mistakes_can_link_to_trade(conn: sqlite3.Connection) -> None:
    cur = conn.execute(
        "INSERT INTO trades(ts, ticker, structure) VALUES (?, ?, ?)",
        (_now(), "NVDA", "long_call"),
    )
    trade_id = cur.lastrowid
    conn.execute(
        "INSERT INTO mistakes(ts, description, lesson, linked_trade_id) "
        "VALUES (?, ?, ?, ?)",
        (_now(), "held through earnings", "size down catalysts", trade_id),
    )
    row = conn.execute(
        "SELECT linked_trade_id FROM mistakes WHERE linked_trade_id = ?",
        (trade_id,),
    ).fetchone()
    assert row["linked_trade_id"] == trade_id


# ─────────────────────────────────────────────────────────────────────────
#  Read-only connection refuses writes
# ─────────────────────────────────────────────────────────────────────────

def test_read_only_connection_refuses_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "ep.sqlite"
    writer = initialize(db_path)
    writer.close()

    reader = get_connection(db_path, read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        reader.execute(
            "INSERT INTO trades(ts, ticker, structure) VALUES (?, ?, ?)",
            (_now(), "X", "long_call"),
        )
    reader.close()


# ─────────────────────────────────────────────────────────────────────────
#  Migration discovery rejects bad filenames
# ─────────────────────────────────────────────────────────────────────────

def test_apply_migrations_rejects_bad_filename(tmp_path: Path) -> None:
    (tmp_path / "garbage.sql").write_text("SELECT 1;")
    conn = get_connection(":memory:")
    with pytest.raises(MigrationError, match="NNN_description"):
        apply_migrations(conn, migrations_dir=tmp_path)
