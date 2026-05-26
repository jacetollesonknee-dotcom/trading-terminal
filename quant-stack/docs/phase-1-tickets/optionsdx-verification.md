# Phase 1 ticket — OptionsDX coverage spot-check

**Status:** open · blocks Phase 2 backtester signal work
**Owner:** operator
**Linked ADR:** [ADR-001](../adr/001-free-data-sources.md)

## Why

ADR-001 commits to OptionsDX free EOD chains as the historical options data
source. Before we backfill ~10 tickers × 5 years into `data/options/` and
build signals against it, we have to confirm OptionsDX actually delivers what
the ADR assumes. If coverage is patchy or Greeks are missing for the
underlyings we care about, we either fall back to capture-forward-only (and
slip Phase 4 signal validation by ~6 months) or escalate to a paid vendor
earlier than planned.

## Acceptance criteria — all must pass to keep ADR-001 in force

### 1. Watchlist coverage

For every ticker in the current watchlist (`memory/structured/watchlists.yaml`,
seeded with NVDA, MU, CSCO, DELL, AVGO, CEG, BWXT, KMI, CTRA, AMD or whatever
Cowork has synced), OptionsDX must provide:

- [ ] Daily EOD chains for **every trading day** from at least 2022-01-01
      through the most recent month-end.
- [ ] Missing-day count < 2% per ticker.
- [ ] Symbol mapping is stable across the window (no ticker renames lost).

### 2. Field completeness

Per contract row, the following must be present and non-null in **≥ 95%** of
samples (1000-row stratified sample per ticker):

- [ ] `bid`, `ask` — never both zero unless contract is delisted.
- [ ] `implied_volatility` — required for IV-rank signals.
- [ ] `delta`, `gamma`, `theta`, `vega` — Greeks.
- [ ] `volume`, `open_interest` — liquidity gates.
- [ ] `expiration`, `strike`, `right` (call/put) — keys.

### 3. Spread sanity

- [ ] `ask >= bid` for ≥ 99.9% of rows. The remaining 0.1% can be flagged at
      ingestion and dropped; report the worst offenders.
- [ ] Bid-ask spread / mid distribution: median < 5% for ATM options on
      large-cap names (NVDA, AAPL-tier liquidity). Wide spreads on thin
      strikes are fine and expected.

### 4. Greeks self-consistency

On a sample of 50 ATM options per ticker:

- [ ] Black-Scholes recomputation of delta from (S, K, r, T, σ=IV) within
      ±0.05 of OptionsDX delta.
- [ ] Put-call parity holds at the IV level within ±2 vol points.

### 5. File-format stability

- [ ] Column names identical across at least the 2022, 2024, and most-recent
      month samples. (Vendor format drift would silently break backfills.)
- [ ] CSV decoding is unambiguous (UTF-8, consistent date format).

### 6. Bulk-download mechanics

- [ ] Full backfill for the watchlist downloads in < 6 hours on the trading
      machine (this is a one-time pull; the ongoing daily delta is small).
- [ ] Download script is rerunnable / idempotent — re-downloading a date
      that already exists locally is a no-op or a checksum match.

## Outputs

When this ticket is closed, write a short report at
`docs/phase-1-tickets/optionsdx-verification-report.md` covering:

- Per-ticker coverage table
- Per-field completeness percentages
- Greeks-recomputation residuals
- Any vendor-side issues found
- Go / no-go recommendation against the criteria above

## Decision rules

- **All criteria pass** → proceed with OptionsDX backfill as ADR-001 specifies.
- **Field completeness < 95% but coverage and spread sanity OK** → proceed,
  but signals that depend on the missing fields are deferred until a paid
  vendor is brought online.
- **Coverage gaps > 2% on any tier-1 watchlist name** → escalate. Either
  fall back to capture-forward-only (β path from earlier discussion) or
  bring forward a paid-vendor evaluation. Update ADR-001 with the decision.
- **Vendor format drifted across years** → fix at ingestion time with a
  per-year adapter; do not silently coerce.

## Out of scope here

- Tick-level / intraday options data. We've committed to EOD for v1.
- Equity historical data — already covered by Schwab + Yahoo, not in this ticket.
- Vendor-comparison work against ORATS or CBOE. That happens only if this
  ticket's criteria fail or NAV crosses the upgrade trigger in ADR-001.
