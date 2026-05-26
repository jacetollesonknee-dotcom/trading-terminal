# ADR-003 — Terminal Flask app integrated under quant-stack/terminal/

**Date:** 2026-05-25
**Status:** Accepted

## Context

Before quant-stack existed, an upgraded Flask + SocketIO trading terminal
was sitting at the repo root (main.py, ai_engine.py, data_feeds.py,
templates/, requirements.txt). It runs `claude-opus-4-7` with streaming
chat, hosts a 5-pass deep-analysis pipeline, watchlist, alerts, real
technicals — works standalone today.

Per the §15 Q1 answer, this existing app is the "terminal template" the
brief's §13 refers to. Per the operator's instruction ("use quant stack in
the new terminal include it"), the right move is to put both inside one
repo so the engine and the terminal evolve together.

## Decision

Move the Flask app inside `quant-stack/terminal/`:

```
quant-stack/
├── config/  ingestion/  memory/  tests/     ← engine
├── docs/
└── terminal/                                ← Flask + SocketIO UI shell
    ├── main.py
    ├── ai_engine.py
    ├── data_feeds.py
    ├── templates/index.html
    ├── requirements.txt
    ├── .env.example
    └── README.md
```

All moves done with `git mv` so history is preserved. The terminal app keeps
working standalone — `cd quant-stack/terminal && python main.py`.

## Consequences

**Accepted:**

- One repo, one history. Engine and terminal evolve in lockstep.
- The terminal's `[terminal]` extra in `pyproject.toml` is the canonical dep
  source. `requirements.txt` remains as a frozen mirror for the standalone
  workflow until Phase 7.5.
- Engine **does not call** anything in `terminal/`. The terminal **does not
  call** anything in the engine yet either — that wiring happens in Phase 7.5.
  Today they coexist; tomorrow the terminal's data layer fetches from the
  engine.

**Quality gates split:**

- `terminal/` is **excluded from** `ruff`, `mypy --strict`, and `bandit` in
  `pyproject.toml`. Reason: pre-existing prototype code without strict type
  coverage. Forcing it through `mypy --strict` would waste a day on noise.
- Engine code (`config/`, `ingestion/`, `memory/`, future
  `models/signals/backtest/risk/execution/`) still passes all strict gates.
- Phase 7.5 cleanup: when the terminal is refactored to fetch from the
  engine's API, it gets folded into the strict gates.

**LLM coupling, scoped:**

- The terminal calls Anthropic directly today (its own copilot panel). That
  is intentional and consistent with ADR-002: **the engine** is headless.
  The terminal is a UI; UIs can call LLMs.
- In Phase 7.5 when the terminal becomes thin (state from engine + render),
  the Anthropic dependency in the terminal can be deleted entirely if the
  Cowork copilot has taken over that surface.

## Folder also reorganized at the repo root

- `main.py`, `ai_engine.py`, `data_feeds.py`, `templates/`,
  `requirements.txt`, `.env.example` — moved into `quant-stack/terminal/`.
- `PUSH_TO_GITHUB.bat`, `SETUP.md`, `.gitignore`, `schwab_trader/` —
  untouched at the repo root.
- `schwab_trader/` is an older duplicate of the Flask app. Leaving it in
  place; safe to delete later if confirmed obsolete.

## Open

Not blocking, but worth re-asking at Phase 7.5:

1. Promote `quant-stack/` to be the actual repo root (move CI workflow up,
   delete the legacy root files)?
2. Delete `schwab_trader/` legacy folder?
