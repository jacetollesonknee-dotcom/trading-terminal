"""Load + validate structured YAML memory.

Every read goes through here. If a YAML file doesn't validate against its
schema, the load fails loudly — no caller ever sees a half-formed dict.

Writes are intentionally absent. To change structured memory, write to
``memory/proposed_updates/`` (see ADR-002), have the operator review the
diff, then commit the YAML edit by hand. Principle #12.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_HERE: Final[Path] = Path(__file__).parent
_STRUCTURED_DIR: Final[Path] = _HERE / "structured"
_SCHEMA_DIR: Final[Path] = _STRUCTURED_DIR / "_schemas"

# Registry of valid memory files. Anything not on this list cannot be loaded.
MEMORY_FILES: Final[dict[str, str]] = {
    "trader_profile": "trader_profile.schema.json",
    "watchlists":     "watchlists.schema.json",
    "strategies":     "strategies.schema.json",
    "greek_targets":  "greek_targets.schema.json",
    "rules":          "rules.schema.json",
    "glossary":       "glossary.schema.json",
}


class MemoryFileNotFoundError(FileNotFoundError):
    """YAML file doesn't exist on disk — usually means it hasn't been seeded yet."""


class MemorySchemaError(ValueError):
    """YAML failed schema validation. Message includes the failing path."""


def _load_schema(name: str) -> dict[str, Any]:
    """Read and parse one JSON Schema. Cached implicitly by validator reuse."""
    if name not in MEMORY_FILES:
        msg = f"unknown memory file {name!r}. valid: {sorted(MEMORY_FILES)}"
        raise KeyError(msg)
    schema_path = _SCHEMA_DIR / MEMORY_FILES[name]
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def load_structured(name: str, *, base_dir: Path | None = None) -> Any:
    """Load and validate a YAML file from memory/structured/.

    :param name: file stem from :data:`MEMORY_FILES` (e.g. ``"strategies"``).
    :param base_dir: override for tests. Default is ``memory/structured/``.
    :returns: the deserialized YAML — usually a dict, sometimes a list
        (strategies, rules, glossary).
    :raises KeyError: ``name`` not in :data:`MEMORY_FILES`.
    :raises MemoryFileNotFoundError: file not on disk.
    :raises MemorySchemaError: file present but doesn't validate.
    """
    schema = _load_schema(name)
    yaml_path = (base_dir or _STRUCTURED_DIR) / f"{name}.yaml"
    if not yaml_path.exists():
        msg = (
            f"{yaml_path} not found. Seed it from the brief or via a "
            "propose_memory_update review."
        )
        raise MemoryFileNotFoundError(msg)
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        Draft202012Validator(schema).validate(data)
    except ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        msg = f"{name}.yaml invalid at {path}: {e.message}"
        raise MemorySchemaError(msg) from e
    return data


def validate_payload(name: str, payload: Any) -> None:
    """Validate an in-memory payload against the named schema.

    Used by the MCP ``propose_memory_update`` tool to check a proposed diff
    before staging it. Raises :class:`MemorySchemaError` on failure.
    """
    schema = _load_schema(name)
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        msg = f"{name} payload invalid at {path}: {e.message}"
        raise MemorySchemaError(msg) from e


def list_available_files() -> list[str]:
    """Return the supported memory-file names."""
    return sorted(MEMORY_FILES)
