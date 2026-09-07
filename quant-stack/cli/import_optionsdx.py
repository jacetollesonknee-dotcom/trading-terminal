"""`python -m cli import-optionsdx DIR --underlying QQQ` — backfill option history.

Imports every OptionsDX EOD CSV under DIR into the store, one snapshot per
quote date, and prints the field-completeness report the Phase 1
verification ticket asks for. Re-running is safe: days already in the
store are reported as deduplicated.

Exit codes: 0 all files imported, 1 some files failed, 2 nothing to import.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import get_settings
from ingestion.optionsdx import import_optionsdx_dir
from storage.store import ParquetStore

# Phase 1 ticket, criterion 2: per-field non-null rate must be at least this.
_TICKET_TARGET = 0.95


def add_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "import-optionsdx",
        help="Backfill option-chain history from OptionsDX EOD CSV files.",
    )
    p.add_argument("directory", help="Folder containing the downloaded .csv/.txt files")
    p.add_argument("--underlying", required=True, help="Ticker the files are for, e.g. QQQ")
    p.add_argument(
        "--iv-scale",
        type=float,
        default=1.0,
        help="Multiply IV by this. OptionsDX quotes a fraction (1.0); use 0.01 for percent.",
    )


def dispatch(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = ParquetStore(settings.data_dir, env=settings.app_env.value)
    try:
        results, summary = import_optionsdx_dir(
            Path(args.directory), store, underlying=args.underlying.upper(),
            iv_scale=args.iv_scale,
        )
    except FileNotFoundError as e:
        print(f"nothing to import: {e}")
        return 2

    print(f"{'file':<40} {'days':>5} {'new':>6} {'dedup':>6}  detail")
    print("-" * 78)
    for r in results:
        name = Path(r.path).name[:40]
        if r.ok:
            print(f"{name:<40} {r.n_snapshots:>5} {r.n_persisted:>6} {r.n_deduplicated:>6}  "
                  f"{r.report.n_contracts:,} contracts, {r.report.n_dropped} dropped")
        else:
            print(f"{name:<40} {'-':>5} {'-':>6} {'-':>6}  FAILED: {r.error}")

    print()
    print("Completeness (Phase 1 ticket, criterion 2 — target >= 95% per field):")
    for line in summary.as_lines():
        print(f"  {line}")
    if summary.open_interest < _TICKET_TARGET:
        print()
        print("  NOTE: open interest is mostly absent. GEX cannot be computed on these")
        print("  years; it will only be available on chains recorded by capture-chains.")
    return 0 if all(r.ok for r in results) else 1
