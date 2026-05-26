# quant-stack

Headless quantitative trading engine for single-operator US equity options.
Naked-only. Paper-then-live. Exposes capabilities to Cowork via MCP.

> Governed by the project brief at `docs/brief-v3.md`. Deviations recorded in `docs/adr/`.

---

## Mission

(see §1 of the brief)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Cowork (LLM, browser, watchlists, copilot UI)          │
└────────────────────────┬────────────────────────────────┘
                         │ MCP (Phase 6.5)
┌────────────────────────▼────────────────────────────────┐
│  quant-stack — headless engine                          │
│   ingestion → models → signals → risk → execution       │
│   structured memory (YAML) + episodic memory (SQLite)   │
└────────────────────────┬────────────────────────────────┘
                         │ httpx
┌────────────────────────▼────────────────────────────────┐
│  Schwab API (live), OptionsDX (historical chains)       │
└─────────────────────────────────────────────────────────┘
```

This engine never calls an LLM. Cowork drives the copilot; the engine answers
tool calls and runs the math.

## Setup

```bash
# 1. Python 3.11
# 2. uv (https://docs.astral.sh/uv/)
uv sync --all-extras
uv run pre-commit install

# 3. One-time secret seeding (uses OS keychain — never .env)
uv run python -m config.secrets --set-schwab-token
```

## Paper-vs-live boundary

`APP_ENV` controls runtime mode:

- `dev` → all writes go to `data/dev/`, no Schwab calls.
- `paper` → real Schwab data, simulated fills, no live orders.
- `live` → live orders **only** with both `APP_ENV=live` and the `--i-mean-it`
  boot flag. Phase 7+ only.

## Terminal-vs-Cowork boundary

- **Cowork** = discretionary cockpit. LLM, browser, watchlists, journals.
- **quant-stack** (this repo) = rules-based engine. No LLM, no browser, no UI of its own.
- **Terminal app** at repo root (Flask + SocketIO) = the §13 UI shell; refactored
  in Phase 7.5 to fetch state from this engine instead of calling Schwab directly.

## Phase status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Foundation, scaffolding, CI, secrets, ingestion interfaces | **in progress** |
| 1 | Data layer (Schwab + OptionsDX historical + capture-forward) | pending |
| 2 | Backtester (event-driven, options-aware) | pending |
| 3 | Models (vol surface, vol forecast, regime, pricing) | pending |
| 4 | Signals (naked-only baseline set) | pending |
| 4.5 | Memory subsystem (structured YAML + episodic SQLite) | pending |
| 5 | Risk + sizing | pending |
| 6 | Paper trading (≥60 trading days) | pending |
| 6.5 | MCP server (engine exposes tools to Cowork) | pending |
| 7 | Live, small | gated |
| 7.5 | Terminal integration | gated |
| 8 | Spread permissions upgrade | **gated, no timeline** |

## Operator runbook

(filled in at Phase 6)

## Non-negotiable principles

See §2 of the brief. Hard rules:
- No look-ahead bias. `as_of` on every record.
- Naked-only — `long_call`, `long_put`, `cash_secured_put`, `covered_call`. Schema- and code-enforced.
- Paper for ≥60 trading days before any live capital.
- No silent fallbacks. No `except: pass`.
- Tax-lot aware. Wash-sale window tracked.
- The engine never routes orders without human confirmation.
- Structured memory never overwritten by code — only via staged diffs.

## Layout

```
quant-stack/
├── pyproject.toml
├── README.md
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── config/                  # settings + OS keychain wrappers
├── ingestion/               # Schwab client + ingested-record schemas
├── memory/                  # structured YAML (schemas in _schemas/) + episodic SQLite
│   ├── structured/_schemas/ # JSON Schema for each YAML file
│   ├── migrations/          # numbered .sql files for episodic.db
│   ├── proposed_updates/    # staged memory diffs awaiting human accept
│   ├── loader.py            # YAML load + jsonschema validate
│   └── episodic.py          # SQLite connection + migration runner
├── terminal/                # Flask + SocketIO UI shell (see ADR-003)
├── data/                    # parquet store, gitignored
├── docs/
│   ├── adr/                 # architecture decision records
│   └── phase-1-tickets/     # pre-implementation acceptance criteria
└── tests/
    └── unit/
```

Future directories (added in their phases): `models/`, `signals/`, `backtest/`,
`risk/`, `execution/`, `reporting/`, `monitoring/`, `mcp_server/`.

## ADRs (architecture decisions)

| # | Title | Status |
|---|-------|--------|
| 001 | Free-only data sources for v1 | accepted |
| 002 | Engine is a tool-server; Cowork owns the LLM | accepted |
| 003 | Terminal Flask app integrated under quant-stack/terminal/ | accepted |
