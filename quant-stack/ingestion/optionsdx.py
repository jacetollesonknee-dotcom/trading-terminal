"""OptionsDX EOD chain importer — the historical backfill ADR-001 chose.

OptionsDX ships one *wide* CSV per underlying per period: one row per
(quote date, expiry, strike) with the call's fields prefixed ``C_`` and the
put's ``P_``, headers wrapped in square brackets with stray whitespace
(``[QUOTE_DATE]``, `` [C_BID]``). This module turns those rows into one
:class:`~ingestion.schema.OptionsChainSnapshot` per quote date and writes
them through the same store the live recorder uses, so backfilled history
and captured-forward history are indistinguishable downstream.

Rules, per the Phase 1 verification ticket:

- **No silent coercion.** A file missing a required column fails with the
  column named. Vendor format drift is fixed here with an explicit adapter,
  never papered over.
- **Crossed quotes are dropped and counted**, not repaired.
- **IV scale is declared, not guessed.** OptionsDX quotes IV as a fraction
  (0.35). If a file's IV looks like percent (median > 3), import stops and
  says so; pass ``iv_scale=0.01`` deliberately.
- **Idempotent.** The store keeps one snapshot per day; re-importing a
  file reports deduplicated, never duplicates.
- **Measure, don't assume.** :func:`completeness` reports the per-field
  non-null rates the ticket asks for. Open interest in particular may be
  absent from OptionsDX files; if it is, GEX cannot be computed on the
  backfilled years, and the report will say so.

``as_of`` is 21:00 UTC on the quote date — the same end-of-session
convention the Yahoo daily bars use.
"""

from __future__ import annotations

import csv
import io
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from ingestion.schema import OptionContract, OptionRight, OptionsChainSnapshot
from storage.store import ParquetStore

# End-of-session stamp for an EOD row (matches the Yahoo daily-bar convention).
_EOD_UTC: Final[time] = time(21, 0)

# A median IV above this can only be percent, not a fraction.
_IV_LOOKS_LIKE_PERCENT: Final[float] = 3.0

_REQUIRED: Final[tuple[str, ...]] = (
    "QUOTE_DATE", "UNDERLYING_LAST", "EXPIRE_DATE", "STRIKE",
    "C_BID", "C_ASK", "P_BID", "P_ASK",
)

# Optional per-side fields, normalized name -> schema attribute.
_SIDE_OPTIONAL: Final[dict[str, str]] = {
    "LAST": "last", "IV": "iv", "DELTA": "delta", "GAMMA": "gamma",
    "THETA": "theta", "VEGA": "vega", "RHO": "rho", "VOLUME": "volume",
}
# Open interest has appeared under different names; accept any, prefer first.
_OI_NAMES: Final[tuple[str, ...]] = ("OI", "OPEN_INTEREST", "OPENINTEREST")


class OptionsDXFormatError(ValueError):
    """The file is not in the shape this adapter understands."""


def normalize_header(name: str) -> str:
    """``" [C_BID]"`` -> ``"C_BID"``."""
    return name.strip().strip("[]").strip().upper()


def _num(value: str | None) -> float | None:
    if value is None:
        return None
    v = value.strip()
    if v == "" or v.upper() in {"NAN", "NULL", "NONE"}:
        return None
    return float(v)


def _date(value: str) -> date:
    v = value.strip()
    # OptionsDX uses YYYY-MM-DD; tolerate a trailing time component.
    return date.fromisoformat(v[:10])


@dataclass
class ParseReport:
    """What one file yielded."""

    path: str
    n_rows: int = 0
    n_contracts: int = 0
    n_dropped: int = 0
    quote_dates: list[date] = field(default_factory=list)
    has_open_interest: bool = False


def _side_contract(
    row: dict[str, str],
    prefix: str,
    right: OptionRight,
    *,
    underlying: str,
    as_of: datetime,
    expiration: date,
    strike: float,
    iv_scale: float,
    oi_col: str | None,
) -> OptionContract | None:
    """Build one side's contract; None if that side has no market at all."""
    bid = _num(row.get(f"{prefix}_BID"))
    ask = _num(row.get(f"{prefix}_ASK"))
    if bid is None and ask is None:
        return None
    kwargs: dict[str, object] = {
        "as_of": as_of, "underlying": underlying, "expiration": expiration,
        "strike": strike, "right": right,
        "bid": max(bid or 0.0, 0.0), "ask": max(ask or 0.0, 0.0), "source": "optionsdx",
    }
    for col, attr in _SIDE_OPTIONAL.items():
        raw = _num(row.get(f"{prefix}_{col}"))
        if raw is None:
            continue
        if attr == "volume":
            kwargs[attr] = int(raw)
        elif attr == "iv":
            kwargs[attr] = raw * iv_scale if raw >= 0.0 else None
        elif attr == "last":
            kwargs[attr] = raw if raw > 0.0 else None
        elif attr == "delta":
            kwargs[attr] = max(-1.0, min(1.0, raw))
        elif attr == "gamma":
            kwargs[attr] = raw if raw >= 0.0 else None
        else:
            kwargs[attr] = raw
    if oi_col is not None:
        oi = _num(row.get(f"{prefix}_{oi_col}"))
        if oi is not None:
            kwargs["open_interest"] = int(oi)
    return OptionContract(**kwargs)


def _row_keys(row: dict[str, str]) -> tuple[date, date, float, float] | None:
    """(quote date, expiration, strike, spot) for a row, or None if unusable."""
    try:
        quote_date = _date(row["QUOTE_DATE"])
        expiration = _date(row["EXPIRE_DATE"])
        strike = float(row["STRIKE"])
        spot = _num(row.get("UNDERLYING_LAST"))
    except (ValueError, KeyError):
        return None
    if spot is None or spot <= 0.0 or strike <= 0.0:
        return None
    return quote_date, expiration, strike, spot


def parse_optionsdx(
    text: str,
    *,
    underlying: str,
    iv_scale: float = 1.0,
    path: str = "<text>",
) -> tuple[list[OptionsChainSnapshot], ParseReport]:
    """Parse one OptionsDX CSV into snapshots, one per quote date, ascending.

    Raises:
        OptionsDXFormatError: Required columns missing, or IV looks like
            percent while ``iv_scale`` says fraction.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        msg = f"{path}: empty file"
        raise OptionsDXFormatError(msg)
    rename = {raw: normalize_header(raw) for raw in reader.fieldnames}
    present = set(rename.values())
    missing = [c for c in _REQUIRED if c not in present]
    if missing:
        msg = f"{path}: missing required column(s) {missing}; found {sorted(present)}"
        raise OptionsDXFormatError(msg)
    oi_col = next((n for n in _OI_NAMES if f"C_{n}" in present or f"P_{n}" in present), None)

    report = ParseReport(path=path, has_open_interest=oi_col is not None)
    by_day: dict[date, tuple[float, list[OptionContract]]] = {}
    ivs: list[float] = []
    for raw_row in reader:
        row = {rename[k]: (v or "") for k, v in raw_row.items() if k is not None}
        report.n_rows += 1
        keys = _row_keys(row)
        if keys is None:
            report.n_dropped += 2  # both sides of an unparseable row
            continue
        quote_date, expiration, strike, spot = keys
        as_of = datetime.combine(quote_date, _EOD_UTC, tzinfo=UTC)
        contracts = by_day.setdefault(quote_date, (spot, []))[1]
        for prefix, right in (("C", OptionRight.call), ("P", OptionRight.put)):
            try:
                c = _side_contract(
                    row, prefix, right, underlying=underlying, as_of=as_of,
                    expiration=expiration, strike=strike, iv_scale=iv_scale, oi_col=oi_col,
                )
            except (ValidationError, ValueError, TypeError):
                report.n_dropped += 1  # crossed quote, bad number — dropped and counted
                continue
            if c is None:
                continue
            contracts.append(c)
            if c.iv is not None:
                ivs.append(c.iv)

    if ivs and statistics.median(ivs) > _IV_LOOKS_LIKE_PERCENT:
        msg = (
            f"{path}: median IV is {statistics.median(ivs):.1f} — that is percent, not a "
            f"fraction. Re-run with iv_scale=0.01 if this file quotes IV in percent."
        )
        raise OptionsDXFormatError(msg)

    snapshots: list[OptionsChainSnapshot] = []
    for quote_date in sorted(by_day):
        spot, contracts = by_day[quote_date]
        if not contracts:
            continue
        snapshots.append(
            OptionsChainSnapshot(
                as_of=datetime.combine(quote_date, _EOD_UTC, tzinfo=UTC),
                underlying=underlying,
                underlying_price=spot,
                contracts=tuple(contracts),
                source="optionsdx",
            )
        )
        report.n_contracts += len(contracts)
        report.quote_dates.append(quote_date)
    return snapshots, report


@dataclass(frozen=True)
class Completeness:
    """Per-field non-null rates across contracts — the ticket's criterion 2."""

    n_contracts: int
    n_days: int
    first_day: date | None
    last_day: date | None
    two_sided_quote: float
    iv: float
    delta: float
    gamma: float
    open_interest: float
    crossed_dropped: int

    def as_lines(self) -> list[str]:
        span = f"{self.first_day} -> {self.last_day}" if self.first_day else "n/a"
        return [
            f"days {self.n_days}  ({span})   contracts {self.n_contracts:,}",
            f"two-sided quote {self.two_sided_quote:6.1%}   iv {self.iv:6.1%}   "
            f"delta {self.delta:6.1%}   gamma {self.gamma:6.1%}   "
            f"open interest {self.open_interest:6.1%}",
            f"crossed / malformed rows dropped: {self.crossed_dropped}",
        ]


def completeness(
    snapshots: Sequence[OptionsChainSnapshot], *, dropped: int = 0
) -> Completeness:
    """Field-completeness over a set of snapshots."""
    contracts = [c for s in snapshots for c in s.contracts]
    n = len(contracts)

    def rate(pred: Iterable[bool]) -> float:
        return (sum(1 for p in pred if p) / n) if n else 0.0

    days = sorted({s.as_of.date() for s in snapshots})
    return Completeness(
        n_contracts=n,
        n_days=len(days),
        first_day=days[0] if days else None,
        last_day=days[-1] if days else None,
        two_sided_quote=rate(c.bid > 0.0 and c.ask > 0.0 for c in contracts),
        iv=rate(c.iv is not None for c in contracts),
        delta=rate(c.delta is not None for c in contracts),
        gamma=rate(c.gamma is not None for c in contracts),
        open_interest=rate(c.open_interest > 0 for c in contracts),
        crossed_dropped=dropped,
    )


@dataclass(frozen=True)
class ImportResult:
    """Outcome of importing one file into the store."""

    path: str
    n_snapshots: int
    n_persisted: int
    n_deduplicated: int
    report: ParseReport
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def import_optionsdx_file(
    path: Path, store: ParquetStore, *, underlying: str, iv_scale: float = 1.0
) -> tuple[ImportResult, list[OptionsChainSnapshot]]:
    """Parse one file and write every snapshot. Format errors are reported, not raised."""
    try:
        text = path.read_text(encoding="utf-8-sig")
        snapshots, report = parse_optionsdx(
            text, underlying=underlying, iv_scale=iv_scale, path=str(path)
        )
    except (OptionsDXFormatError, OSError, UnicodeDecodeError) as e:
        return (
            ImportResult(str(path), 0, 0, 0, ParseReport(path=str(path)), error=str(e)),
            [],
        )
    persisted = dedup = 0
    for snap in snapshots:
        w = store.write_options_chain(snap)
        persisted += w.persisted
        dedup += w.deduplicated
    return ImportResult(str(path), len(snapshots), persisted, dedup, report), snapshots


def import_optionsdx_dir(
    directory: Path, store: ParquetStore, *, underlying: str, iv_scale: float = 1.0
) -> tuple[list[ImportResult], Completeness]:
    """Import every ``*.csv`` / ``*.txt`` under ``directory`` (sorted) and summarize.

    Raises:
        FileNotFoundError: No matching files.
    """
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".csv", ".txt"})
    if not files:
        msg = f"no .csv/.txt files under {directory}"
        raise FileNotFoundError(msg)
    results: list[ImportResult] = []
    all_snaps: list[OptionsChainSnapshot] = []
    dropped = 0
    for f in files:
        result, snaps = import_optionsdx_file(f, store, underlying=underlying, iv_scale=iv_scale)
        results.append(result)
        all_snaps.extend(snaps)
        dropped += result.report.n_dropped
    return results, completeness(all_snaps, dropped=dropped)
