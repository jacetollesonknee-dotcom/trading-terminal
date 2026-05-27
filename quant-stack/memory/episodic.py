"""Episodic memory: append-only SQLite event log.

Six tables (see ``migrations/001_init.sql``):

- ``trades``           -- every closed paper/live trade
- ``conversations``    -- every Cowork agent turn that reached this engine
- ``positions``        -- per-reconciliation snapshot of the book
- ``signals_fired``    -- every signal score + decision
- ``mistakes``         -- post-mortems, linked to trades
- ``proposed_updates`` -- staged diffs against structured memory

This module owns connection lifecycle and migration application. It does NOT
own write APIs for individual tables — those live next to the domain logic
that uses them (e.g. ``signals/base.py`` will write to ``signals_fired``).

Threading: SQLite is happy with multiple readers and one writer per
connection. The MCP server holds one writer connection; ad-hoc readers open
their own.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Final

_HERE: Final[Path] = Path(__file__).parent
_MIGRATIONS_DIR: Final[Path] = _HERE / "migrations"


class MigrationError(RuntimeError):
    """A migration file failed to apply or was skipped out of order."""


def get_connection(db_path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection with sensible defaults.

    - Foreign keys ON.
    - WAL journal mode (concurrent readers + one writer).
    - Row factory returns ``sqlite3.Row`` so columns are name-addressable.

    :param db_path: filesystem path to the .sqlite file. ``":memory:"`` works for tests.
    :param read_only: if True, opens via URI in ``mode=ro`` so writes raise.
    """
    if read_only and db_path != ":memory:":
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only and db_path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _discover_migrations(migrations_dir: Path) -> list[tuple[int, Path]]:
    """Return migrations as (version_int, path), sorted ascending.

    Filename convention: ``NNN_description.sql`` where NNN is zero-padded.
    """
    out: list[tuple[int, Path]] = []
    pattern = re.compile(r"^(\d{3,})_[a-z0-9_]+\.sql$")
    for p in sorted(migrations_dir.glob("*.sql")):
        m = pattern.match(p.name)
        if not m:
            msg = f"migration {p.name} does not match NNN_description.sql"
            raise MigrationError(msg)
        out.append((int(m.group(1)), p))
    versions = [v for v, _ in out]
    if versions != sorted(set(versions)):
        msg = f"migration versions must be unique and sequential: {versions}"
        raise MigrationError(msg)
    return out


def _applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Read versions previously applied. Empty set on a fresh DB."""
    try:
        rows = conn.execute("SELECT version FROM schema_versions").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {int(r["version"]) for r in rows}


def apply_migrations(
    conn: sqlite3.Connection, *, migrations_dir: Path | None = None
) -> list[int]:
    """Apply any unapplied migrations from ``migrations_dir``.

    Returns the list of version numbers actually applied (empty if up to date).

    Each migration runs in its own transaction. On failure, the transaction
    rolls back and :class:`MigrationError` propagates — the DB is left at
    the last successfully-applied version.
    """
    migrations = _discover_migrations(migrations_dir or _MIGRATIONS_DIR)
    already = _applied_versions(conn)
    applied: list[int] = []
    for version, path in migrations:
        if version in already:
            continue
        sql = path.read_text(encoding="utf-8")
        try:
            with conn:  # transaction
                conn.executescript(sql)
        except sqlite3.Error as e:
            msg = f"migration {path.name} failed: {e}"
            raise MigrationError(msg) from e
        applied.append(version)
    return applied


def initialize(db_path: Path | str) -> sqlite3.Connection:
    """Convenience: open ``db_path`` and apply any pending migrations.

    Returns a writer connection ready for use.
    """
    conn = get_connection(db_path)
    apply_migrations(conn)
    return conn
