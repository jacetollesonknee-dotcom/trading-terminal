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
├── backtest/                # vectorized engine + validation stack (see below)
├── options/                 # GEX analytics + naked-only options simulator
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

Future directories (added in their phases): `models/`, `signals/`,
`risk/`, `execution/`, `reporting/`, `monitoring/`, `mcp_server/`.

## Backtesting and validation (`backtest/`, `options/`)

Two engines, one ledger shape, one validation stack.

- **`backtest.run_backtest`** — vectorized, look-ahead-safe, prices a *linear*
  position in an underlying. Fast signal research.
- **`options.simulate`** — event-driven over chain snapshots, **naked-only
  enforced in code** (long call, long put, cash-secured put, covered call).
  Fills at bid/ask, marks at mid, settles at expiry. This is the Phase 2
  options backtester.

Both emit the same ledger, so metrics, deflated Sharpe, regime attribution,
and the health monitor apply to either.

**Options quickstart** (synthetic chains until your Schwab capture fills the store):

```bash
cd quant-stack
uv sync --all-extras                                    # once
uv run python examples/gex_naked_options.py --synthetic
```

```python
from options import Book, ContractKey, SimConfig, gex_profile, simulate

def strategy(chain, pf):                   # chain at as_of + Portfolio(book, cash, equity)
    prof = gex_profile(chain)              # net GEX, call/put walls, gamma flip
    if prof.dealer_gamma == "long" and not pf.book.options and pf.can_secure_puts(prof.put_wall):
        return pf.book.with_option(ContractKey(exp, prof.put_wall, OptionRight.put), -1)  # CSP
    return pf.book

sim = simulate(chain_snapshots, strategy, SimConfig(initial_capital=50_000))
sim.result.metrics.sharpe     # same PerformanceMetrics as the vectorized engine
sim.fills, sim.settlements    # every execution and expiry, for reconciliation
```

A naked short call, or a short put without its strike notional in cash,
raises `NakedOnlyError` and stops the run — the code-level twin of the
schema's `AllowedStructure` gate.

**Spreads** are a research switch, off by default:

```python
SimConfig(initial_capital=25_000, allow_spreads=True)
```

On, a short leg may be covered by a long leg of the same right expiring on or
after it — verticals, iron condors, butterflies, calendars, diagonals. The
collateral is the defined risk (strike width for a credit spread; nothing
extra for a debit spread); exceeding cash raises `CollateralError`. A naked
short call is refused either way. This switch changes what you can *test*.
The live-order schema stays naked-only until the Phase 8 spread-permissions
gate is opened on purpose.

**Vectorized engine** for a linear position:

```python
from backtest import BacktestConfig, run_backtest
# prices: pd.Series of closes, ascending UTC DatetimeIndex, strictly positive.
# signal: target position in [-1, 1], SAME index, computed from data available
#         AT that timestamp.
result = run_backtest(prices, signal, BacktestConfig())   # 252 periods/yr default
result.metrics.sharpe, result.metrics.max_drawdown, result.equity_curve
```

Design notes:

- **No look-ahead.** The signal at bar *t* is acted on at *t + execution_lag*
  (lag ≥ 1, enforced by the config). Principle #1 of the brief, verified by a
  hypothesis property test in `tests/property/test_backtest_lookahead.py`.
- **No silent fallbacks.** Misaligned indices, non-monotonic time, duplicate
  timestamps, and non-positive prices are hard errors, not quiet repairs. A
  `NaN` in the *signal* is the one tolerated gap — it means "flat".
- **Costs.** Turnover (`|Δposition|`) is charged in bps of notional (fee +
  slippage), plus an optional per-bar carry on the held position. The options
  simulator instead pays the real spread (buy at ask, sell at bid) plus
  per-contract commission — there is no slippage knob to set to zero.

### Deflated Sharpe (`backtest/deflated_sharpe.py`)

If you tested N variations and kept the best, its Sharpe is inflated by
selection. `deflated_sharpe` (Bailey & López de Prado 2014) gives the
probability the observed Sharpe beats the best-of-N-noise expectation:

```python
from backtest import deannualize_sharpe, deflated_sharpe

trials = [deannualize_sharpe(m.sharpe, 365) for m in every_variation_you_ran]
res = deflated_sharpe(result.ledger["net"], trial_sharpes=trials)
res.deflated_sharpe, res.passes   # probability, and >= 0.95 by default
```

Everything is **per-period, never annualized** — the Sharpe and moments are
derived from the return series itself so the units can't be mixed up, and
`SR*` is scaled by the dispersion of the trials as the paper requires. Be
honest about `trial_sharpes`: its length and spread are what deflate the result.

### Walk-forward (`backtest/walk_forward.py`)

Fit on the past, trade the unseen future, roll:

```python
from backtest import FittedStrategy, WalkForwardConfig, walk_forward

def fit(train: pd.Series) -> FittedStrategy:
    lookback = pick_lookback(train)                     # sees train ONLY
    def signal(history: pd.Series) -> pd.Series:        # sees history through the test window
        ma = history.rolling(lookback).mean()
        return (history > ma).astype(float).where(ma.notna())
    return FittedStrategy(signal_fn=signal, params={"lookback": lookback})

res = walk_forward(prices, fit, BacktestConfig(), WalkForwardConfig(train_bars=180, test_bars=60))
res.result.metrics.sharpe   # headline: ONE backtest over the stitched OOS span
res.fold_metrics            # per-fold diagnostics (noisy — consistency, not results)
res.param_table             # fitted params per fold; instability = overfitting tell
```

Two departures from the textbook loop: the signal function receives
contiguous history *through* the test window so lookback indicators aren't
cold-started every fold (the engine's `execution_lag` guards the future, not
the slice), and out-of-sample signals are stitched into a single backtest so
positions carry across fold boundaries instead of restarting flat each fold.

### Position sizing (`backtest/sizing.py`)

Fixed-fractional sizing with an honest stop:

```python
from backtest import Side, SizingConfig, position_size

ps = position_size(capital=10_000, entry=100, stop=95, side=Side.long,
                   cfg=SizingConfig(risk_fraction=0.01, max_position_fraction=0.20,
                                    stop_slippage_bps=3, fee_bps=5, lot_step=0.001))
ps.position_fraction   # signed notional/capital — drop it straight into a signal
ps.loss_if_stopped     # includes slippage on the stop fill and fees on both legs
ps.capped, ps.floored, ps.below_min_notional   # each one means realized risk < target
```

The loss budget includes the stop filling worse than the stop price and fees
on both legs, so `risk_fraction` is what you'd *actually* lose. A long with
its stop above entry (or a short with it below) is rejected as a fat-finger.
Quantity is floored to `lot_step` — never rounded up past the budget — and
zeroed below `min_notional` rather than returned as an untradeable number.

### Regime attribution (`backtest/regimes.py`)

Where did the Sharpe come from?

```python
from backtest import RegimeConfig, regime_report

rep = regime_report(result, prices, RegimeConfig(ma_bars=200, chop_band=0.02))
rep.table                 # per-regime Sharpe, return, hit rate, time share, P&L share
rep.single_regime_edge    # True if only one regime carries a positive Sharpe
rep.verdict()             # "Edge exists ONLY in bull (41% of time, 112% of P&L)..."
```

Labels are trailing (no look-ahead). Drawdown is deliberately not reported
per regime — the bars aren't contiguous.

### Health monitor (`backtest/monitor.py`)

Run on a schedule. It **recommends**; it never acts.

```python
from backtest import Benchmark, HealthConfig, health_check

bench = Benchmark.from_result(walk_forward_result.result)   # OOS, not in-sample
rep = health_check(live_net_returns, bench, HealthConfig(window=30, consecutive_required=3))
rep.alerts            # SHARPE_DECAY / DRAWDOWN_EXCEEDED / UNDERWATER_TOO_LONG / NO_ACTIVITY
rep.recommendation    # insufficient_data | continue | review | halt_recommended
rep.live_sharpe_se    # read the live Sharpe with this beside it
```

A 30-bar annualized Sharpe has a standard error of ~3.5; a "halt below half
the backtest Sharpe" rule fires on ~40% of days for a strategy whose true
Sharpe is 1.5. The monitor instead asks how likely the window is *given* the
benchmark Sharpe (PSR), requires the breach to persist, and treats a window
with no trades as a different alert from one that's losing.

### DSR hurdle (`minimum_sharpe_to_pass`)

Before running anything: how good must the best of N look to be
distinguishable from noise?

```python
from backtest import minimum_sharpe_to_pass
minimum_sharpe_to_pass(n_obs=4380, n_trials=50)   # per-period Sharpe floor
```

## ADRs (architecture decisions)

| # | Title | Status |
|---|-------|--------|
| 001 | Free-only data sources for v1 | accepted |
| 002 | Engine is a tool-server; Cowork owns the LLM | accepted |
| 003 | Terminal Flask app integrated under quant-stack/terminal/ | accepted |
