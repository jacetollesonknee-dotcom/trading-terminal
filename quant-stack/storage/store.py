"""Parquet writer with as_of enforcement and append-or-upsert semantics.

Invariants enforced by this module:

1. Every record carries a non-null UTC ``as_of``. Pydantic enforces this
   on construction; we re-check at the parquet boundary as defense in depth.
2. Append-or-upsert by a record-type-specific primary key. An existing key
   on disk is silently dropped from the new batch — never overwritten.
3. Schema mismatch refuses to write. PyArrow validates against the explicit
   schemas declared in this module before bytes touch disk.

Concurrency: per-file flock (filelock library, cross-platform). Atomic
write via temp-then-rename so a crash mid-write leaves the prior good
file intact.

Compression: zstd. Good ratio, fast read/write.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from ingestion.schema import (
    CorporateAction,
    EarningsEvent,
    EquityBar,
    OptionContract,
    OptionsChainSnapshot,
)
from storage import partition

# ─────────────────────────────────────────────────────────────────────────────
#  Schemas — explicit, never inferred
# ─────────────────────────────────────────────────────────────────────────────

_EQUITY_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("as_of", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("interval", pa.string(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.int64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

_OPTION_CONTRACT_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("as_of", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("underlying", pa.string(), nullable=False),
        # underlying_price is denormalized from the OptionsChainSnapshot it
        # belongs to. Same value repeated for every contract in the snapshot.
        pa.field("underlying_price", pa.float64(), nullable=False),
        pa.field("expiration", pa.date32(), nullable=False),
        pa.field("strike", pa.float64(), nullable=False),
        pa.field("right", pa.string(), nullable=False),
        pa.field("bid", pa.float64(), nullable=False),
        pa.field("ask", pa.float64(), nullable=False),
        pa.field("last", pa.float64(), nullable=True),
        pa.field("volume", pa.int64(), nullable=False),
        pa.field("open_interest", pa.int64(), nullable=False),
        pa.field("iv", pa.float64(), nullable=True),
        pa.field("delta", pa.float64(), nullable=True),
        pa.field("gamma", pa.float64(), nullable=True),
        pa.field("theta", pa.float64(), nullable=True),
        pa.field("vega", pa.float64(), nullable=True),
        pa.field("rho", pa.float64(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
    ]
)

_CORP_ACTION_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("as_of", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("ex_date", pa.date32(), nullable=False),
        pa.field("type", pa.string(), nullable=False),
        pa.field("ratio", pa.float64(), nullable=True),
        pa.field("cash_amount", pa.float64(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
    ]
)

_EARNINGS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("as_of", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("report_date", pa.date32(), nullable=False),
        pa.field("when", pa.string(), nullable=False),
        pa.field("eps_estimate", pa.float64(), nullable=True),
        pa.field("eps_actual", pa.float64(), nullable=True),
        pa.field("eps_surprise_pct", pa.float64(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
    ]
)

_COMPRESSION: Final[str] = "zstd"
_COMPRESSION_LEVEL: Final[int] = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Public types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome of a single write call.

    Sum identity: ``persisted + deduplicated + rejected_schema == requested``.
    """

    requested: int
    persisted: int
    deduplicated: int
    rejected_schema: int
    files_touched: tuple[Path, ...] = field(default_factory=tuple)


class StoreError(RuntimeError):
    """Raised when the store cannot safely complete a write."""


# ─────────────────────────────────────────────────────────────────────────────
#  Store
# ─────────────────────────────────────────────────────────────────────────────


class ParquetStore:
    """Writes immutable records to a partitioned parquet store.

    Bound to ``(base_dir, env)`` at construction. Other envs require a
    different instance. Cheap to construct; safe to keep one per process.
    """

    def __init__(self, base_dir: Path, env: str) -> None:
        if not env:
            msg = "env must be a non-empty string"
            raise ValueError(msg)
        self._base = Path(base_dir)
        self._env = env
        self._base.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base

    @property
    def env(self) -> str:
        return self._env

    # ── public API ────────────────────────────────────────────────────────

    def write_equity_bars(self, bars: Iterable[EquityBar]) -> WriteResult:
        """Persist a batch of equity bars.

        Bars are grouped by (symbol, year, interval) and written to one
        parquet file per group. Existing (symbol, as_of, interval) keys
        on disk are dropped from the batch.
        """
        bars_list = list(bars)
        if not bars_list:
            return WriteResult(0, 0, 0, 0)

        groups: dict[tuple[str, int, str], list[EquityBar]] = {}
        for b in bars_list:
            key = (b.symbol.upper(), b.as_of.year, b.interval)
            groups.setdefault(key, []).append(b)

        total_persisted = 0
        total_dedup = 0
        files: list[Path] = []
        for (symbol, year, interval), batch in groups.items():
            path = partition.equity_file(self._base, self._env, symbol, year, interval)
            persisted, dedup = self._upsert(
                path,
                batch,
                _EQUITY_SCHEMA,
                _equity_to_record,
                key_cols=("as_of", "symbol", "interval"),
            )
            total_persisted += persisted
            total_dedup += dedup
            if persisted:
                files.append(path)

        return WriteResult(
            requested=len(bars_list),
            persisted=total_persisted,
            deduplicated=total_dedup,
            rejected_schema=0,
            files_touched=tuple(files),
        )

    def write_options_chain(self, snapshot: OptionsChainSnapshot) -> WriteResult:
        """Persist an options chain snapshot.

        One file per (underlying, expiry, snapshot_date). If the file
        already exists, every contract in this batch is counted as
        deduplicated — we do not overwrite snapshots.
        """
        if not snapshot.contracts:
            return WriteResult(0, 0, 0, 0)
        snapshot_date = snapshot.as_of.date()

        by_expiry: dict[date, list[OptionContract]] = {}
        for c in snapshot.contracts:
            by_expiry.setdefault(c.expiration, []).append(c)

        total_persisted = 0
        total_dedup = 0
        files: list[Path] = []
        for expiry, contracts in by_expiry.items():
            path = partition.options_chain_file(
                self._base, self._env, snapshot.underlying, expiry, snapshot_date
            )
            if path.exists():
                total_dedup += len(contracts)
                continue
            self._write_new_options(
                path,
                contracts,
                underlying_price=snapshot.underlying_price,
            )
            total_persisted += len(contracts)
            files.append(path)

        return WriteResult(
            requested=len(snapshot.contracts),
            persisted=total_persisted,
            deduplicated=total_dedup,
            rejected_schema=0,
            files_touched=tuple(files),
        )

    def write_corporate_actions(self, actions: Iterable[CorporateAction]) -> WriteResult:
        actions_list = list(actions)
        if not actions_list:
            return WriteResult(0, 0, 0, 0)
        by_symbol: dict[str, list[CorporateAction]] = {}
        for a in actions_list:
            by_symbol.setdefault(a.symbol.upper(), []).append(a)
        total_persisted = 0
        total_dedup = 0
        files: list[Path] = []
        for sym, batch in by_symbol.items():
            path = partition.corporate_actions_file(self._base, self._env, sym)
            persisted, dedup = self._upsert(
                path,
                batch,
                _CORP_ACTION_SCHEMA,
                _corp_action_to_record,
                key_cols=("symbol", "ex_date", "type"),
            )
            total_persisted += persisted
            total_dedup += dedup
            if persisted:
                files.append(path)
        return WriteResult(
            requested=len(actions_list),
            persisted=total_persisted,
            deduplicated=total_dedup,
            rejected_schema=0,
            files_touched=tuple(files),
        )

    def write_earnings(self, events: Iterable[EarningsEvent]) -> WriteResult:
        events_list = list(events)
        if not events_list:
            return WriteResult(0, 0, 0, 0)
        by_symbol: dict[str, list[EarningsEvent]] = {}
        for e in events_list:
            by_symbol.setdefault(e.symbol.upper(), []).append(e)
        total_persisted = 0
        total_dedup = 0
        files: list[Path] = []
        for sym, batch in by_symbol.items():
            path = partition.earnings_file(self._base, self._env, sym)
            persisted, dedup = self._upsert(
                path,
                batch,
                _EARNINGS_SCHEMA,
                _earnings_to_record,
                key_cols=("symbol", "report_date"),
            )
            total_persisted += persisted
            total_dedup += dedup
            if persisted:
                files.append(path)
        return WriteResult(
            requested=len(events_list),
            persisted=total_persisted,
            deduplicated=total_dedup,
            rejected_schema=0,
            files_touched=tuple(files),
        )

    # ── internals ─────────────────────────────────────────────────────────

    def _upsert(
        self,
        path: Path,
        records: list[Any],
        schema: pa.Schema,
        to_record: Callable[[Any], dict[str, Any]],
        *,
        key_cols: tuple[str, ...],
    ) -> tuple[int, int]:
        """Append records to ``path``; drop rows whose key already exists.

        Returns ``(persisted_count, deduplicated_count)``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path) + ".lock")
        with lock:
            existing_keys: set[tuple[Any, ...]] = set()
            if path.exists():
                existing_keys = _read_existing_keys(path, key_cols)

            new_dicts: list[dict[str, Any]] = []
            dedup_count = 0
            for r in records:
                d = to_record(r)
                k = tuple(d[c] for c in key_cols)
                if k in existing_keys:
                    dedup_count += 1
                    continue
                existing_keys.add(k)  # in-batch dedup too
                new_dicts.append(d)

            if not new_dicts:
                return 0, dedup_count

            new_table = pa.Table.from_pylist(new_dicts, schema=schema)
            if path.exists():
                existing_table = pq.read_table(path, schema=schema)
                combined = pa.concat_tables([existing_table, new_table])
            else:
                combined = new_table

            self._atomic_write(combined, path)
            return len(new_dicts), dedup_count

    def _write_new_options(
        self,
        path: Path,
        contracts: list[OptionContract],
        *,
        underlying_price: float,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path) + ".lock")
        with lock:
            if path.exists():
                msg = f"refusing to overwrite existing snapshot at {path}"
                raise StoreError(msg)
            records = [_option_to_record(c, underlying_price=underlying_price) for c in contracts]
            table = pa.Table.from_pylist(records, schema=_OPTION_CONTRACT_SCHEMA)
            self._atomic_write(table, path)

    @staticmethod
    def _atomic_write(table: pa.Table, path: Path) -> None:
        """Write to a temp file then atomically rename. Safe across crashes."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(
            table,
            tmp,
            compression=_COMPRESSION,
            compression_level=_COMPRESSION_LEVEL,
        )
        tmp.replace(path)


def _read_existing_keys(path: Path, key_cols: tuple[str, ...]) -> set[tuple[Any, ...]]:
    """Read only the key columns from an existing file. Cheap dedup check."""
    table = pq.read_table(path, columns=list(key_cols))
    cols = [table.column(c).to_pylist() for c in key_cols]
    return set(zip(*cols, strict=True))


# ─────────────────────────────────────────────────────────────────────────────
#  pydantic → dict adapters (one per record type)
# ─────────────────────────────────────────────────────────────────────────────


def _equity_to_record(b: EquityBar) -> dict[str, Any]:
    return {
        "as_of": b.as_of,
        "symbol": b.symbol.upper(),
        "interval": b.interval,
        "open": b.open,
        "high": b.high,
        "low": b.low,
        "close": b.close,
        "volume": b.volume,
        "source": b.source,
    }


def _option_to_record(c: OptionContract, *, underlying_price: float) -> dict[str, Any]:
    return {
        "as_of": c.as_of,
        "underlying": c.underlying.upper(),
        "underlying_price": underlying_price,
        "expiration": c.expiration,
        "strike": c.strike,
        "right": c.right.value,
        "bid": c.bid,
        "ask": c.ask,
        "last": c.last,
        "volume": c.volume,
        "open_interest": c.open_interest,
        "iv": c.iv,
        "delta": c.delta,
        "gamma": c.gamma,
        "theta": c.theta,
        "vega": c.vega,
        "rho": c.rho,
        "source": c.source,
    }


def _corp_action_to_record(a: CorporateAction) -> dict[str, Any]:
    return {
        "as_of": a.as_of,
        "symbol": a.symbol.upper(),
        "ex_date": a.ex_date,
        "type": a.type,
        "ratio": a.ratio,
        "cash_amount": a.cash_amount,
        "source": a.source,
    }


def _earnings_to_record(e: EarningsEvent) -> dict[str, Any]:
    return {
        "as_of": e.as_of,
        "symbol": e.symbol.upper(),
        "report_date": e.report_date,
        "when": e.when,
        "eps_estimate": e.eps_estimate,
        "eps_actual": e.eps_actual,
        "eps_surprise_pct": e.eps_surprise_pct,
        "source": e.source,
    }
