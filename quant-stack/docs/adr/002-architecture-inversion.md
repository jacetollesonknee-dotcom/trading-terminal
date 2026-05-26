# ADR-002 — Engine is a tool-server; Cowork owns the LLM

**Date:** 2026-05-25
**Status:** Accepted (supersedes Part B §11, §12, §13 of the brief)

## Context

The brief's Part B specified an LLM copilot layer inside this project: an
agent loop calling Anthropic, a system-prompt assembled from structured
memory, token budgets, ingestion of Cowork notes into a semantic vector store,
and an `<AgentPanel />` in the terminal UI.

Operator clarified that Cowork already has all of that — LLM access, browser
(Claude in Chrome), watchlist sync, the copilot UI. The right shape for this
project is the inverse: a headless quant engine that **exposes** capabilities
to Cowork via an MCP server.

## Decision

This project never calls an LLM and never instantiates an embedding model.

| Brief section | Status |
|---|---|
| §10.1 Structured memory (YAML) | **kept** |
| §10.2 Semantic memory (Chroma + embeddings) | **dropped** |
| §10.3 Episodic memory (SQLite) | **kept** |
| §11 Cowork import pipeline | **dropped** (data flows the other way) |
| §12 LLM agent layer | **dropped** |
| §13 `<AgentPanel />` in terminal | **dropped** |
| §6.5 build agent layer | **replaced** by "build MCP server" |
| §7.5 terminal integration | kept, refactored to fetch from MCP-backed engine |
| Phase 0 item 7 (ANTHROPIC keychain) | **dropped** |
| Phase 0 items 12–14 (FastAPI + agent) | **dropped** (Phase 6.5 builds MCP, not FastAPI) |

Cowork uses our MCP tools the same way it currently uses its own Schwab MCP,
ToS bridge, and Claude-in-Chrome. The engine's job is to be a reliable tool
provider with hard guardrails on what it will execute.

## Consequences

**Accepted:**
- Drastically smaller surface area in this repo. No agent loop, no token
  budgets, no embeddings, no semantic store, no Anthropic SDK dependency.
- Cost of running this engine is dominated by Schwab API + disk; no per-call LLM cost.
- Single LLM context lives in Cowork — no risk of drift between two copilots.

**Constraints we still inherit from the brief:**
- §2 Principle #11 — agent (in Cowork) proposes; human executes. Enforced
  here by the order-routing endpoint refusing non-confirmed orders.
- §2 Principle #12 — structured memory only mutated via staged diffs the
  operator accepts. Enforced here by the MCP `propose_memory_update` tool
  writing to `memory/proposed_updates/` and never to `memory/structured/`.
- §2 Principle #7 — naked-only. Enforced at schema layer (JSON Schema rejects
  disallowed `structure` values) and at the order generator.

## Open Phase 6.5 question

MCP-only, or MCP **plus** a thin FastAPI HTTP surface for scripts and the
terminal's read-only panels? Leaning MCP-only; revisit at Phase 6.5 start.
