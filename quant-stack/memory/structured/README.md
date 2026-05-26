# Structured memory

YAML files in this directory are the source of truth for facts the engine and
Cowork's agent must not forget. Schemas in `_schemas/` validate every load.

| File | Schema | Purpose |
|------|--------|---------|
| `trader_profile.yaml` | `trader_profile.schema.json` | NAV, account, options permissions, existing book, goals |
| `watchlists.yaml` | `watchlists.schema.json` | Tier + thesis overlay on top of Yahoo-synced symbols |
| `strategies.yaml` | `strategies.schema.json` | Named strategies. **Schema enforces naked-only** until Phase 8 |
| `greek_targets.yaml` | `greek_targets.schema.json` | Portfolio + per-position risk limits, drawdown breakers |
| `rules.yaml` | `rules.schema.json` | Machine-readable mirror of §2 non-negotiables |
| `glossary.yaml` | `glossary.schema.json` | Personal terminology |

## Rules

- **Git-tracked.** Every change is a commit.
- **Schema-validated on every load.** `memory.loader.load_structured()` refuses
  invalid YAML before it ever reaches a caller.
- **Never mutated by code in this repo.** Edits go through
  `propose_memory_update` → `memory/proposed_updates/` → human accept → manual
  commit. Principle #12.
- **Yaml not JSON** because humans edit these files. JSON Schema validates
  them but the on-disk format is the lossless YAML.

## Seeding

These six files are intentionally **not** committed at Phase 0 — they hold your
personal trader profile and would leak details into git. Seed them locally
from the brief once you confirm the schemas:

```bash
# example
cat > memory/structured/trader_profile.yaml <<'EOF'
nav_band:
  current_usd: 19500
  target_milestones:
    - {date: 2026-06-01, nav_usd: 25000}
    - {date: 2026-10-01, nav_usd: 50000}
account: {broker: schwab, type: personal_taxable_margin, pdt_flagged: true}
options_permissions:
  level: 2
  allowed_structures: [long_call, long_put, cash_secured_put, covered_call]
  forbidden_personal: [naked_short_call]
existing_book:
  - {symbol: NVDA, type: equity, qty: 200, cost_basis: 181.40}
  - {symbol: NVDA, type: covered_call, qty: -2, expiration: 2025-06-18, strike: 220}
margin: {current_debt_usd: 21000, cap_pct_of_nav: 30}
EOF

python -c "from memory.loader import load_structured; load_structured('trader_profile')"
# raises with a precise error if anything is off; succeeds silently if valid.
```
