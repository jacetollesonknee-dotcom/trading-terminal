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
_INSIDER_TRADES: Final[str] = "insider_trades"
_ANALYST_RATINGS: Final[str] = "analyst_ratings"
_SOCIAL_POSTS: Final[str] = "social_posts"


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


def insider_trades_file(base_dir: Path, env: str, ticker: str, year: int) -> Path:
    """One file per (ticker, filing-year) for insider trades."""
    return (
        base_dir / env / _INSIDER_TRADES / ticker.upper()
        / f"year={year}" / "trades.parquet"
    )


def insider_trades_symbol_root(base_dir: Path, env: str, ticker: str) -> Path:
    """Root of all years for one ticker's insider trades."""
    return base_dir / env / _INSIDER_TRADES / ticker.upper()


def analyst_ratings_file(base_dir: Path, env: str, symbol: str, year: int) -> Path:
    """One file per (symbol, year) for analyst ratings."""
    return (
        base_dir / env / _ANALYST_RATINGS / symbol.upper()
        / f"year={year}" / "ratings.parquet"
    )


def analyst_ratings_symbol_root(base_dir: Path, env: str, symbol: str) -> Path:
    """Root of all years for one symbol's analyst ratings."""
    return base_dir / env / _ANALYST_RATINGS / symbol.upper()


def social_posts_file(
    base_dir: Path, env: str, platform: str, author_handle: str, year: int
) -> Path:
    """One file per (platform, author, year) for social posts."""
    safe = _sanitise_handle(author_handle)
    return (
        base_dir / env / _SOCIAL_POSTS / platform.lower() / safe
        / f"year={year}" / "posts.parquet"
    )


def social_posts_author_root(
    base_dir: Path, env: str, platform: str, author_handle: str
) -> Path:
    """Root of all years for one author's posts on one platform."""
    safe = _sanitise_handle(author_handle)
    return base_dir / env / _SOCIAL_POSTS / platform.lower() / safe


def social_posts_platform_root(base_dir: Path, env: str, platform: str) -> Path:
    """Root of all authors for one platform."""
    return base_dir / env / _SOCIAL_POSTS / platform.lower()


def _sanitise_handle(handle: str) -> str:
    """Make a handle safe for the filesystem. Strips '@' and uppercases."""
    return handle.lstrip("@").upper()


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


def scan_insider_trades_files(base_dir: Path, env: str) -> list[Path]:
    root = base_dir / env / _INSIDER_TRADES
    if not root.exists():
        return []
    return sorted(root.rglob("trades.parquet"))


def scan_analyst_ratings_files(base_dir: Path, env: str) -> list[Path]:
    root = base_dir / env / _ANALYST_RATINGS
    if not root.exists():
        return []
    return sorted(root.rglob("ratings.parquet"))


def scan_social_posts_files(base_dir: Path, env: str) -> list[Path]:
    root = base_dir / env / _SOCIAL_POSTS
    if not root.exists():
        return []
    return sorted(root.rglob("posts.parquet"))
