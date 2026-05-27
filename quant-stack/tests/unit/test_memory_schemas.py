"""Schema-level tests for memory/structured/_schemas/*.schema.json.

Two flavors:

1. Every schema is itself a valid JSON Schema (Draft 2020-12).
2. The naked-only allowlist is enforced on the right surfaces:
   - strategies.yaml — `structure` must be in the allowlist.
   - trader_profile.yaml — `options_permissions.allowed_structures` ditto.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml as pyyaml
from jsonschema import Draft202012Validator

from memory.loader import (
    MEMORY_FILES,
    MemoryFileNotFoundError,
    MemorySchemaError,
    list_available_files,
    load_structured,
    validate_payload,
)

_SCHEMA_DIR = Path(__file__).parents[2] / "memory" / "structured" / "_schemas"

ALLOWED_STRUCTURES = ("long_call", "long_put", "cash_secured_put", "covered_call")
DISALLOWED_STRUCTURES = ("vertical", "iron_condor", "naked_short_call",
                         "short_strangle", "calendar", "diagonal")


# ─────────────────────────────────────────────────────────────────────────
#  Every schema is itself valid
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("schema_file", sorted(_SCHEMA_DIR.glob("*.schema.json")))
def test_each_schema_is_a_valid_jsonschema(schema_file: Path) -> None:
    with schema_file.open() as f:
        schema = json.load(f)
    # Raises if the schema itself is malformed.
    Draft202012Validator.check_schema(schema)


def test_loader_registry_matches_disk() -> None:
    on_disk = {p.stem.replace(".schema", "") for p in _SCHEMA_DIR.glob("*.schema.json")}
    assert set(MEMORY_FILES) == on_disk
    assert list_available_files() == sorted(MEMORY_FILES)


# ─────────────────────────────────────────────────────────────────────────
#  Naked-only enforcement — strategies.yaml
# ─────────────────────────────────────────────────────────────────────────

def _valid_strategy(structure: str = "cash_secured_put") -> dict[str, Any]:
    return {
        "name": "ai_infra_premium_harvest_csp",
        "thesis": "Sell CSPs on AI infra names with IV rank > 60 in low-vol regime.",
        "underlyings": ["NVDA"],
        "structure": structure,
        "entry": {"iv_rank_min": 60},
        "exit": {"profit_target_pct": 50},
        "position_sizing": {"max_buying_power_pct": 5, "kelly_fraction": 0.25},
    }


@pytest.mark.parametrize("structure", ALLOWED_STRUCTURES)
def test_strategies_accept_allowed_structures(structure: str) -> None:
    validate_payload("strategies", [_valid_strategy(structure)])


@pytest.mark.parametrize("structure", DISALLOWED_STRUCTURES)
def test_strategies_reject_disallowed_structures(structure: str) -> None:
    with pytest.raises(MemorySchemaError, match="structure"):
        validate_payload("strategies", [_valid_strategy(structure)])


def test_strategies_reject_unknown_top_level_field() -> None:
    bad = _valid_strategy()
    bad["secret_sauce"] = "lol"
    with pytest.raises(MemorySchemaError):
        validate_payload("strategies", [bad])


def test_strategies_require_thesis_of_substance() -> None:
    """30-char minimum on thesis — discourages drive-by entries."""
    s = _valid_strategy()
    s["thesis"] = "vibes"
    with pytest.raises(MemorySchemaError):
        validate_payload("strategies", [s])


# ─────────────────────────────────────────────────────────────────────────
#  Naked-only enforcement — trader_profile.yaml allowed_structures
# ─────────────────────────────────────────────────────────────────────────

def _valid_trader_profile() -> dict[str, Any]:
    return {
        "nav_band": {"current_usd": 19500},
        "account": {"broker": "schwab", "type": "personal_taxable_margin", "pdt_flagged": True},
        "options_permissions": {
            "level": 2,
            "allowed_structures": ["long_call", "long_put", "cash_secured_put", "covered_call"],
            "forbidden_personal": ["naked_short_call"],
        },
        "existing_book": [{"symbol": "NVDA", "type": "equity", "qty": 200, "cost_basis": 181.40}],
        "margin": {"current_debt_usd": 21000, "cap_pct_of_nav": 30},
    }


def test_trader_profile_round_trips() -> None:
    validate_payload("trader_profile", _valid_trader_profile())


@pytest.mark.parametrize("structure", DISALLOWED_STRUCTURES)
def test_trader_profile_rejects_disallowed_in_permissions(structure: str) -> None:
    p = _valid_trader_profile()
    p["options_permissions"]["allowed_structures"].append(structure)
    with pytest.raises(MemorySchemaError):
        validate_payload("trader_profile", p)


# ─────────────────────────────────────────────────────────────────────────
#  Loader file-not-found behavior
# ─────────────────────────────────────────────────────────────────────────

def test_load_structured_raises_when_missing(tmp_path: Path) -> None:
    """Loader fails loudly when YAML is missing — no silent empty dict."""
    with pytest.raises(MemoryFileNotFoundError, match="not found"):
        load_structured("strategies", base_dir=tmp_path)


def test_load_structured_round_trips_a_real_file(tmp_path: Path) -> None:
    (tmp_path / "rules.yaml").write_text(
        pyyaml.safe_dump([{
            "id": "naked_only",
            "description": "Only naked structures allowed.",
            "enforced_at": ["schema", "order_generator"],
            "permanent_until": "phase_8",
            "principle_ref": 7,
        }])
    )
    loaded = load_structured("rules", base_dir=tmp_path)
    assert loaded[0]["id"] == "naked_only"


def test_load_structured_rejects_unknown_name() -> None:
    with pytest.raises(KeyError):
        load_structured("does_not_exist")
