"""Partition-path helpers for the on-disk parquet store.

One source of truth for where data lives on disk. Pure helpers, no I/O.

Layout (namespaced by env so dev/paper/live never overlap):

    data/{env}/equities/{SYMBOL}/year={YYYY}/interval={1m|1d|...}/bars.parquet
    data/{env}/options/{UNDERLYING}/expiry={YYYY-MM-DD}/snapshot={YYYY-MM-DD}.parquet
    data/{env}/corporate_actions/{SYMBOL}.parquet
    data/{env}/earnings/{SYMBOL}.parquet
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final

_EQUITIES: Final[str] = "equities"
_OPTIONS: Final[str] = "options"
_CORP_ACTIONS: Final[str] = "corporate_actions"
_EARNINGS: Final[str] = "earnings"


# ─────────────────────────────────────────────────────────────────────────────
#  Path builders
# ─────────────────────────────────────────────────────────────────────────────


def equity_dir(base_dir: Path, env: str, symbol: str, year: int, interval: str) -> Path:
    """Directory containing the parquet file for one (symbol, year, interval)."""
    return (
        base_dir / env / _EQUITIES / symbol.upper()
        / f"year={year}" / f"interval={interval}"
    )


def equity_file(base_dir: Path, env: str, symbol: str, year: int, interval: str) -> Path:
    """Parquet file path for equity bars."""
    return equity_dir(base_dir, env, symbol, year, interval) / "bars.parquet"


def equity_symbol_root(base_dir: Path, env: str, symbol: str) -> Path:
    """Root of all years/intervals for one symbol — useful for recursive globs."""
    return base_dir / env / _EQUITIES / symbol.upper()


def options_chain_file(
    base_dir: Path,
    env: str,
    underlying: str,
    expiry: date,
    snapshot: date,
) -> Path:
    """One file per (underlying, expiry, snapshot_date)."""
    return (
        base_dir / env / _OPTIONS / underlying.upper()
        / f"expiry={expiry.isoformat()}"
        / f"snapshot={snapshot.isoformat()}.parquet"
    )


def options_underlying_root(base_dir: Path, env: str, underlying: str) -> Path:
    """Root for all expiries/snapshots of one underlying."""
    return base_dir / env / _OPTIONS / underlying.upper()


def corporate_actions_file(base_dir: Path, env: str, symbol: str) -> Path:
    """One file per symbol for corporate actions."""
    return base_dir / env / _CORP_ACTIONS / f"{symbol.upper()}.parquet"


def earnings_file(base_dir: Path, env: str, symbol: str) -> Path:
    """One file per symbol for earnings events."""
    return base_dir / env / _EARNINGS / f"{symbol.upper()}.parquet"


# ─────────────────────────────────────────────────────────────────────────────
#  Scanners — return existing parquet paths for DuckDB to read across
# ─────────────────────────────────────────────────────────────────────────────


def scan_equity_files(base_dir: Path, env: str) -> list[Path]:
    """Every existing equity parquet under base_dir/env. Returns [] if missing."""
    root = base_dir / env / _EQUITIES
    if not root.exists():
        return []
    return sorted(root.rglob("bars.parquet"))


def scan_options_files(base_dir: Path, env: str) -> list[Path]:
    """Every existing options chain snapshot. Returns [] if missing."""
    root = base_dir / env / _OPTIONS
    if not root.exists():
        return []
    return sorted(root.rglob("*.parquet"))


def scan_corporate_actions_files(base_dir: Path, env: str) -> list[Path]:
    root = base_dir / env / _CORP_ACTIONS
    if not root.exists():
        return []
    return sorted(root.glob("*.parquet"))


def scan_earnings_files(base_dir: Path, env: str) -> list[Path]:
    root = base_dir / env / _EARNINGS
    if not root.exists():
        return []
    return sorted(root.glob("*.parquet"))
