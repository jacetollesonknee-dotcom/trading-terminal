# ADR-001 — Free-only data sources for v1

**Date:** 2026-05-25
**Status:** Accepted

## Context

The brief at §Phase 1 §Data Sources flagged historical options chains as the
single biggest data risk and recommended subscribing to ORATS (~$50–100/mo) to
unblock realistic backtesting within Phase 2's timeline.

Operator instead chose: free sources only for v1.

## Decision

**Plan α** (free):
- **Equities historical:** Schwab API + Yahoo (already integrated in the terminal layer). 10y+ of clean EOD daily and intraday.
- **Options live:** Schwab API for current chains with Greeks, IV, OI, volume.
- **Options historical:** OptionsDX free daily EOD chain CSVs (~2020–present) bulk backfilled into `data/options/`.
- **Options capture-forward:** nightly Schwab snapshot starting day 1.

## Consequences

**Accepted:**
- Phase 4 signal validation works on OptionsDX history (~5y) plus capture-forward — enough for walk-forward at the watchlist scale.
- All four sources are zero marginal cost.

**Risks:**
- OptionsDX coverage and quality must be verified during Phase 1; if it falls short, fallback is capture-forward only and Phase 4 validation slips ~6 months.
- No tick-level options data. Backtester models fills at bid/ask EOD, not intraday spread evolution.
- If account scales past Phase 7, upgrading to ORATS or CBOE DataShop is a `data/options/` source swap, not a rewrite — designed for that.

## Upgrade trigger

NAV ≥ $50K **or** Phase 7 reveals fill-model error >50bps on EOD chains.
