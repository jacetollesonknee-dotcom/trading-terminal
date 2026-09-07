"""`python -m cli capture-chains ...` — record today's option chains.

    capture-chains QQQ SPY            Snapshot the chains for QQQ and SPY now.
    capture-chains QQQ --broker tos   Use a different enabled broker.

Writes to ``data/<app_env>/options/``. Run it once per trading day after
the close (a cron / Task Scheduler entry is enough), or let the ingestion
scheduler do it with ``chains_enabled=True``. A second run the same day is
reported as deduplicated — snapshots are never overwritten.

Exit codes: 0 all captured, 1 some failed, 2 broker not enabled.
"""

from __future__ import annotations

import argparse

from config.settings import BrokerName, get_settings
from ingestion.brokers.registry import BrokerDisabledError
from ingestion.chain_capture import run_capture
from storage.store import ParquetStore


def add_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "capture-chains",
        help="Record today's option chain snapshot for one or more underlyings.",
    )
    p.add_argument("underlyings", nargs="+", help="Tickers, e.g. QQQ SPY")
    p.add_argument(
        "--broker",
        default=BrokerName.schwab.value,
        choices=[b.value for b in BrokerName],
        help="Enabled broker to pull chains from (default: schwab).",
    )


def dispatch(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = ParquetStore(settings.data_dir, env=settings.app_env.value)
    try:
        results = run_capture(settings, store, args.underlyings, broker_name=args.broker)
    except BrokerDisabledError as e:
        print(f"cannot capture: {e}")
        return 2

    print(f"{'underlying':<12} {'status':<8} {'persisted':>9} {'dedup':>6}  detail")
    print("-" * 60)
    for r in results:
        if r.ok and r.write_result is not None:
            w = r.write_result
            print(f"{r.underlying:<12} {'ok':<8} {w.persisted:>9} {w.deduplicated:>6}  "
                  f"{len(w.files_touched)} file(s)")
        else:
            print(f"{r.underlying:<12} {'FAILED':<8} {'-':>9} {'-':>6}  {r.error}")
    return 0 if all(r.ok for r in results) else 1
