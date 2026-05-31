# ADR-004 — Terminal is the all-in-one app; engine is its backend

**Date:** 2026-05-30
**Status:** Accepted (revises ADR-002)

## Context

ADR-002 cast this project as a headless quant engine that exposes tools to
Cowork's LLM via MCP. The terminal at `quant-stack/terminal/` was the §13
UI shell — a working prototype but not the primary surface.

Operator's revised vision: **one all-in-one trading app**. Live positions,
AI chat that knows the portfolio, real broker connection, news + insider +
analyst + sentiment + signals all in one place. The user opens one thing,
not two.

## Decision

The Flask terminal at `quant-stack/terminal/` becomes the primary product
surface. The engine at `quant-stack/` is the systematic backend the
terminal calls into. Cowork stays available for ad-hoc research but is
no longer a first-class architectural target.

| What | Before | After |
|---|---|---|
| Primary UI | Cowork (for AI) + terminal (for charts) | Terminal — everything |
| LLM home | Cowork | Terminal's AI Chat panel |
| Engine consumer | Cowork via MCP | Terminal via in-process import (or HTTP if we add an API) |
| Cowork | First-class | Optional research cockpit |
| MCP server | Phase 6.5 priority | Still useful for external consumers + Cowork holdout; not the *primary* integration target |
| Brokers connect UI | CLI only | CLI **plus** terminal UI (login modal, status pill) |

## Consequences

**Accepted:**
- Terminal grows into the all-in-one app: real broker connection, real
  positions, proposed-trade-confirmation flow, full signal surfacing.
- The terminal's data layer migrates from "talks to Yahoo/Zacks/OpenInsider
  directly" → "calls into the engine's ingestion + storage." Phase 7.5
  becomes much more important and likely pulls earlier than originally
  scheduled.
- Cowork integration drops in priority. The §1 boundary about "Cowork
  remains the discretionary cockpit" is now a *user preference*, not an
  architectural pillar.
- ADR-002's mappings change:
  - §12 LLM agent layer — **partially un-dropped**. The terminal already
    has an Opus 4.7 chat panel. We do NOT need a separate `backend/agent/`
    in the engine — the terminal owns the agent loop. But the engine
    exposes the tools that loop calls (memory, signals, positions, etc.).
  - §13 `<AgentPanel />` in terminal — **un-dropped**. Already exists.

**Constraints we still inherit from the brief (unchanged):**
- §2 Principle #11 — agent (in terminal) proposes; human confirms
  in-terminal; engine routes. Human-in-the-loop is mandatory.
- §2 Principle #12 — structured memory only mutated via staged diffs.
- §2 Principle #7 — naked-only. Schema- and code-enforced.
- §2 Principle #1 — no look-ahead bias. Already enforced by
  storage/query.py.

## Phase reprioritisation

| Phase | Was | Now |
|---|---|---|
| 1 Data layer | Equity backfill + options chain history | Same, but every ingestion source must surface to the terminal UI |
| 4.5 Memory | Seeded for the Cowork agent | Seeded for the terminal's chat panel — same content |
| 6.5 MCP server | Build to expose engine to Cowork | **Deferred.** Engine ↔ terminal can live in-process (Flask imports engine modules directly). MCP becomes a Phase 8+ "external integrations" concern. |
| 7.5 Terminal refactor | Refactor terminal's data layer to fetch from engine | **Pull earlier.** Phase 5 or sooner. This is now the critical path. |

## What stays from ADR-002

- Engine is still headless from the engine's own perspective. No LLM
  calls *originate* in `quant-stack/` modules. The terminal's `ai_engine.py`
  calls Anthropic; that's the terminal's job. The engine's signals and
  models are pure math.
- Structured memory remains git-tracked, edited only via staged diffs.
- Naked-only enforcement remains at the schema + DB layer.

## Open

1. When the terminal refactors to the engine's data layer (Phase 7.5),
   does the terminal still also keep direct calls to Yahoo/news as a
   fallback when the engine has no data yet? Probably yes for v1, with
   a feature flag.
2. Do we eventually retire `terminal/data_feeds.py` entirely once the
   engine's ingestion is comprehensive? Probably yes in Phase 8+.
