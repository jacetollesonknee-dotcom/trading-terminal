# terminal/ — Flask + SocketIO UI shell

> Integrated under quant-stack per **ADR-003**. This is the §13 terminal layer
> from the original brief: charts, watchlist, news, deep-research panel,
> human-in-the-loop trade ticket.

## What it currently does (today, standalone)

- Live Yahoo Finance / Zacks / OpenInsider / Kalshi quotes and chains
- 9-source news aggregator (CNBC, MarketWatch, WSJ, Benzinga, Seeking Alpha,
  Reuters, Investing.com, Yahoo Finance, plus CNN Fear & Greed)
- Streaming chat against `claude-opus-4-7` (extended thinking + prompt caching)
- 5-pass deep-analysis pipeline (Technical / Fundamental / Sentiment / Risk /
  Master Strategist) streamed to the UI via SSE
- Real technicals: RSI, SMA 10/20/50/200, EMA, MACD with signal, Bollinger,
  ATR, composite score
- Persistent paper portfolio + watchlist + alerts (`trader_state.json`)
- Live price updates via Socket.IO

## How it fits with the engine

Right now: **standalone**. The terminal talks directly to Yahoo, Zacks,
OpenInsider, Anthropic. It does not yet know about the quant engine sitting
next to it.

In Phase 7.5: refactored. The terminal's data layer fetches from the engine
(via the Phase 6.5 MCP server's HTTP companion, or whatever surface the engine
exposes for read-only UI consumption). Single source of truth for state.

In the meantime the two coexist cleanly because:

- The engine is **headless** (no LLM, no UI).
- The terminal is a **UI shell** that happens to do its own data layer.
- They share the repo but not the runtime — each runs on its own.

## Run it standalone

```bash
# from the repo root
cd quant-stack/terminal
cp .env.example .env       # set ANTHROPIC_API_KEY here
pip install -r requirements.txt
python main.py
```

Opens `http://127.0.0.1:5000`.

## Or via the unified pyproject

```bash
cd quant-stack
uv sync --extra terminal   # installs Flask + Anthropic + scrape deps
cd terminal && python main.py
```

`requirements.txt` is kept as a frozen mirror for the standalone path; the
canonical source is `pyproject.toml`'s `[terminal]` extra.

## Engine integration (ADR-004 Phase 7.5)

The terminal consumes the quant-stack engine **in-process** via
`engine_bridge.py`. On boot it starts the engine's `IngestionScheduler`
(yahoo / openinsider / zacks, and x when enabled), which polls each source
on its own cadence, persists deduped records to the point-in-time parquet
store, and streams every poll cycle to the browser as an `ingestion_update`
SocketIO event. The **Sentiment** panel surfaces this live in the *Live
Ingestion* card plus a *Social Buzz* card backed by the engine's stored X
posts.

New surfaces:

| Route | Serves |
|---|---|
| `GET /api/engine/status` | scheduler availability + running state + watchlist |
| `GET /api/sentiment/<sym>` | stored X/social posts for `$sym` (falls back to headlines) |
| `GET /api/insider/<sym>` | engine store first (point-in-time, deduped), live scrape as fallback |
| SocketIO `ingestion_update` | per-poll event stream from the scheduler |

The bridge **degrades gracefully**: if the engine deps aren't importable or
`ENGINE_SCHEDULER=false`, every engine-backed route falls back to the
terminal's own `data_feeds.py` and the app runs unchanged. The reusable,
tested piece of this integration lives in the engine
(`ingestion/serialize.py`) so it's covered by the engine's gates; the
terminal side is thin glue.

## Why this isn't in the engine's strict CI

The terminal is pre-existing prototype code. It doesn't carry `mypy --strict`
type coverage and shouldn't gate engine PRs while it grows up. `terminal/` is
excluded from ruff, mypy, and bandit in `pyproject.toml`. Its own quality
gates (and tests) get added during the Phase 7.5 rewrite.

## What to NOT touch here

- **Order routing.** The terminal proposes; the human confirms; orders go
  through the engine in Phase 7.5. The Flask `/api/trade` endpoint runs
  against the paper portfolio only.
- **Naked-only.** Even in the prototype, the order form should only let you
  enter `long_call`, `long_put`, `cash_secured_put`, `covered_call`. The
  engine's schema enforces this at the API boundary in Phase 7.5.
