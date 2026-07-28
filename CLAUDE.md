# CLAUDE.md

Guidance for AI assistants (Claude Code and others) working in this repository.

## What this project is

A single-operator US-equity **options trading system**, **naked-only**,
**paper-then-live**. It has two cooperating parts that live in one repo:

1. **`quant-stack/` — the engine.** A headless, rules-based quantitative engine:
   ingestion → storage → (future) models → signals → risk → execution. It
   **never calls an LLM**, has no UI of its own, and is held to strict quality
   gates. This is where nearly all the real code lives.
2. **`quant-stack/terminal/` — the app.** A pre-existing Flask + SocketIO
   trading terminal (live quotes, charts, news, an Anthropic-powered AI chat
   panel, paper portfolio). Per **ADR-004** this is the primary product surface;
   the engine is its systematic backend. It is a prototype excluded from the
   engine's strict gates.

The governing intent and every deviation are recorded in `quant-stack/docs/adr/`.
**Read the ADRs before making architectural changes** — the design has been
inverted twice (see ADR-002 → ADR-004) and the ADRs are the source of truth for
current direction, more so than older prose in READMEs.

## Repository layout

```
trading-terminal/                 ← repo root (primary working dir)
├── CLAUDE.md                      ← this file
├── SETUP.md                       ← end-user setup for the terminal app (Schwab + Claude)
├── PUSH_TO_GITHUB.bat            ← legacy helper, ignore
└── quant-stack/                   ← the engine + the terminal; almost all work happens here
    ├── pyproject.toml             ← deps, ruff/mypy/bandit/pytest config (single source of truth)
    ├── uv.lock                    ← pinned lockfile (uv)
    ├── .pre-commit-config.yaml    ← mirror of CI gates (minus pytest)
    ├── .github/workflows/ci.yml   ← CI: ruff + mypy + bandit + pytest on ubuntu & windows
    ├── config/                    ← Settings (pydantic-settings) + OS-keychain secret wrappers
    ├── ingestion/                 ← all data-source clients + the record schemas + scheduler
    │   └── brokers/               ← broker Protocol, registry (feature-gated), Schwab/TOS clients
    ├── storage/                   ← parquet writer + partition paths + point-in-time query layer
    ├── memory/                    ← structured YAML (schema-validated) + episodic SQLite
    │   ├── structured/_schemas/   ← JSON Schema per YAML file
    │   ├── migrations/            ← numbered .sql for episodic.db (never edit an applied one)
    │   └── proposed_updates/      ← staged memory diffs awaiting human accept
    ├── cli/                       ← `python -m cli connect ...` operator CLI
    ├── terminal/                  ← Flask + SocketIO UI shell (excluded from strict gates)
    ├── docs/adr/                  ← architecture decision records (authoritative)
    └── tests/                     ← unit / integration / property
```

Directories added in later phases (do not exist yet): `models/`, `signals/`,
`backtest/`, `risk/`, `execution/`, `reporting/`, `monitoring/`, `mcp_server/`.

`data/` (parquet store) and `logs/`, `audit/` are runtime-created and gitignored.
The `memory/structured/*.yaml` files themselves are **intentionally not
committed** (they hold personal trader data) — only their schemas are tracked.

## Development commands

All commands run from **`quant-stack/`** and use [`uv`](https://docs.astral.sh/uv/).
Python is pinned to `>=3.11,<3.13`.

```bash
cd quant-stack

uv sync --all-extras            # install engine + terminal + dev deps
uv run pre-commit install       # one-time; wires the local gates

# The four quality gates (identical to CI — run all before pushing):
uv run ruff check .             # lint (also: `uv run ruff format` to format)
uv run mypy .                   # strict type-check (config/, ingestion/, memory/, storage/, cli/, tests/)
uv run bandit -r . -ll --exclude tests
uv run pytest -q                # tests (excludes `live`-marked by default)
```

Test markers (`pyproject.toml`): `integration`, `property`, `live`.
`live` tests hit real external APIs and are **skipped by default** (`-m 'not live'`).
Opt in with `uv run pytest -m live`. Run one file: `uv run pytest tests/unit/test_schema.py`.

Run the terminal app standalone:
```bash
cd quant-stack/terminal
cp .env.example .env            # set ANTHROPIC_API_KEY
python main.py                  # → http://127.0.0.1:5000
```

Seed a Schwab OAuth token (goes to the OS keychain, never `.env`):
```bash
uv run python -m config.secrets --set-schwab-token --env production
uv run python -m cli connect status     # inspect broker connection state
```

## Non-negotiable principles (enforced, not aspirational)

These are hard rules. Code and schemas enforce them; do not weaken them.

1. **No look-ahead bias.** Every stored record carries a UTC `as_of`. All reads
   go through `storage/query.py::PointInTimeQuery`, which appends
   `as_of <= decision_time` to every query — callers cannot opt out.
2. **Naked-only.** The only permitted structures are `long_call`, `long_put`,
   `cash_secured_put`, `covered_call`. Enforced at three layers: the
   `AllowedStructure` Literal in `ingestion/schema.py`, a `CHECK` constraint on
   `trades.structure` in `migrations/001_init.sql`, and the order generator.
   Phase 8 (gated, no timeline) is the *only* place this widens, via a new
   migration — never by editing `001_init.sql`.
3. **Records are immutable.** All ingested models set `frozen=True` +
   `extra="forbid"`. To "update" a record, write a new one with a later `as_of`.
4. **Paper before live.** Live orders require **both** `APP_ENV=live` **and** the
   `--i-mean-it` boot flag; `Settings.live_orders_armed` is the gate and the
   order router refuses otherwise. `paper_trading=True` is the default hard switch.
5. **No silent fallbacks. No `except: pass`.** Failures surface loudly. (The one
   deliberate swallow is the scheduler's `_emit` guarding a *consumer's* callback
   — the scheduler itself never fails silently.)
6. **The engine never calls an LLM.** No Anthropic/embedding deps in engine
   modules. The terminal's `ai_engine.py` calls Anthropic — that's the UI's job,
   not the engine's (ADR-002 / ADR-004).
7. **Human-in-the-loop for orders.** The agent/terminal proposes; the human
   confirms; the engine routes. No auto-execution without confirmation.
8. **Structured memory is never mutated by code.** Changes go
   `proposed_updates/` → human review → manual YAML edit + commit (Principle #12).
   `memory/loader.py` is read + validate only; there is intentionally no write API.
9. **UTC at the storage boundary.** Everything is stored UTC-aware; naive
   datetimes are rejected. Presentation layers (terminal) render local time.

## Key conventions

- **Settings** (`config/settings.py`): one frozen `Settings` object from
  `get_settings()`, loaded from `.env` + env vars, `extra="forbid"`. `APP_ENV`
  (`dev`/`paper`/`live`) drives `scoped_data_dir` so `data/dev|paper|live` never
  overlap. Non-sensitive config only.
- **Secrets** (`config/secrets.py`): OS keychain via `keyring`. Tokens never
  touch `.env` or git. Namespaced `quant-stack.schwab.<env>`.
- **Record schemas** (`ingestion/schema.py`): every ingested/stored/routed
  record is a pydantic model here. Adding a data type starts with a schema here,
  then a PyArrow schema + write method in `storage/store.py`, a partition path
  in `storage/partition.py`, and a read method in `storage/query.py`.
- **Storage** (`storage/`): parquet + zstd, explicit (never inferred) PyArrow
  schemas, append-or-upsert by a per-type primary key (existing keys are dropped,
  never overwritten), atomic temp-then-rename writes under a cross-platform
  `filelock`. Reads via DuckDB `read_parquet`.
- **Ingestion clients**: all free public sources for v1 (ADR-001). Sync `httpx`
  through the shared `ingestion/_http.py::get_with_retry` (429 + 5xx retryable
  with backoff; other 4xx permanent). HTML scrapers (Zacks, OpenInsider, X via
  rsshub) are defensive — a missing field is `None`, but any record that *does*
  parse must pass pydantic validation.
- **Scheduler** (`ingestion/scheduler.py`): one daemon thread per source, polled
  (free sources have no push feeds), emits `SchedulerEvent`s. Clients are
  lazy-imported inside each tick so the scheduler works with partial deps.
- **Brokers** (`ingestion/brokers/`): acquire via `registry.get_broker(name,
  settings)` only — it enforces the per-broker `brokers_enabled` feature flag
  (default all-off). Schwab and TOS share the Schwab Trader API; both are Phase-0
  stubs raising `NotImplementedError` with a phase pointer until OAuth lands.
- **Episodic memory** (`memory/episodic.py` + `migrations/`): append-only SQLite,
  WAL mode, numbered `NNN_description.sql` migrations applied in order and
  recorded in `schema_versions`. Never edit an already-applied migration — add a
  new one.

## Quality-gate scope (important gotcha)

`terminal/` is **excluded** from ruff, mypy, and bandit in `pyproject.toml`
(pre-existing prototype; folds into strict gates during the Phase 7.5 rewrite).
Everything else in `quant-stack/` passes `mypy --strict`. Per-file ruff ignores
exist for CLI `print`, keychain CLI, and intentional lazy imports in the registry
and scheduler — check `[tool.ruff.lint.per-file-ignores]` before "fixing" a lint
that's already deliberately suppressed.

## Phase status (from quant-stack/README.md)

Phase 0 (foundation/ingestion interfaces) is in progress; Phase 1 (data layer)
work is landing incrementally (`1.4a`–`1.4e`: brokers scaffold, OpenInsider,
Zacks, X, and the polling scheduler). Backtester, models, signals, risk,
execution, and the MCP server are pending/deferred. Live trading (Phase 7+) and
spread permissions (Phase 8) are gated.

## Git workflow

- Do all work on the designated feature branch; create it locally if missing.
- Commit with clear messages; push with `git push -u origin <branch>` (retry with
  backoff on network errors only).
- `no-commit-to-branch` (pre-commit) blocks direct commits to `main`.
- Do **not** open a PR unless explicitly asked.
- Keep the model identifier out of commit messages, code, and any pushed artifact.
