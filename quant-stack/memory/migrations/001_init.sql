-- 001_init.sql — initial episodic-memory schema for quant-stack.
-- All timestamps are ISO-8601 UTC strings (SQLite has no native datetime).
--
-- Hard rules baked in:
--  * trades.structure is constrained to the naked-only allowlist. Phase 8 will
--    add a new migration to widen this; this file is never edited.
--  * proposed_updates.status is a constrained enum so the review workflow
--    can't silently move into an unknown state.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ── trades ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    ticker        TEXT    NOT NULL,
    structure     TEXT    NOT NULL,
    rationale     TEXT,
    signal_scores TEXT,                          -- JSON blob
    agent_role    TEXT,                          -- e.g. 'cowork-agent' | 'human' | 'auto'
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    CHECK (structure IN ('long_call', 'long_put', 'cash_secured_put', 'covered_call'))
);
CREATE INDEX IF NOT EXISTS idx_trades_ts     ON trades(ts);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);

-- ── conversations ─────────────────────────────────────────────────────────
-- Write-heavy. WITHOUT ROWID keeps the primary index dense and the table small.
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,                -- ULID / UUID7 from caller
    ts          TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool', 'system')),
    content     TEXT,
    tool_calls  TEXT,                            -- JSON blob
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_conv_session_ts ON conversations(session_id, ts);

-- ── positions ─────────────────────────────────────────────────────────────
-- Snapshot per reconciliation. snapshot_json is the full Position[] payload.
CREATE TABLE IF NOT EXISTS positions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_positions_ts ON positions(ts);

-- ── signals_fired ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals_fired (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    signal_name  TEXT NOT NULL,
    score        REAL NOT NULL,
    action_taken TEXT NOT NULL CHECK (action_taken IN
        ('proposed', 'auto_executed', 'rejected_size', 'rejected_risk',
         'rejected_concentration', 'rejected_pdt', 'rejected_margin',
         'rejected_drawdown', 'rejected_other')),
    inputs_json  TEXT,                           -- JSON: what the signal saw
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signals_ts   ON signals_fired(ts);
CREATE INDEX IF NOT EXISTS idx_signals_name ON signals_fired(signal_name);

-- ── mistakes ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS mistakes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    description     TEXT NOT NULL,
    lesson          TEXT,
    linked_trade_id INTEGER REFERENCES trades(id) ON DELETE SET NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_mistakes_ts ON mistakes(ts);

-- ── proposed_updates ──────────────────────────────────────────────────────
-- Staged diffs to memory/structured/*.yaml. The agent writes here; human
-- accepts / rejects / supersedes; only then does anyone touch the YAML.
CREATE TABLE IF NOT EXISTS proposed_updates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    file        TEXT NOT NULL,                   -- e.g. 'strategies.yaml'
    diff        TEXT NOT NULL,                   -- unified diff
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'accepted', 'rejected', 'superseded')),
    proposed_by TEXT,                            -- session_id of the agent turn
    reviewed_by TEXT,
    reviewed_at TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_proposed_status ON proposed_updates(status);
CREATE INDEX IF NOT EXISTS idx_proposed_file   ON proposed_updates(file);

-- ── schema_versions (bookkeeping for the migration runner) ────────────────
CREATE TABLE IF NOT EXISTS schema_versions (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    filename   TEXT NOT NULL
);
INSERT INTO schema_versions(version, filename) VALUES (1, '001_init.sql');
