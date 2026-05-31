"""Point-in-time read layer over the parquet store.

Every read is filtered to ``as_of <= decision_time``. The filter is
applied by the query API; callers cannot opt out. This is where the
brief's Principle #1 (no look-ahead bias) is enforced at the data plane.

Backed by DuckDB reading parquet directly via ``read_parquet``. No
intermediate copy. The ``as_of <= decision_time`` predicate is appended
to every SQL statement issued from this module.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from ingestion.schema import (
    CorporateAction,
    EarningsEvent,
    EquityBar,
    InsiderTrade,
    OptionContract,
    OptionRight,
    OptionsChainSnapshot,
)
from storage import partition
from storage.store import ParquetStore


def _to_posix(p: Path) -> str:
    """DuckDB on Windows accepts forward-slash paths regardless of host."""
    return str(p).replace("\\", "/")


def _ensure_utc(v: Any) -> datetime:
    if not isinstance(v, datetime):
        msg = f"expected datetime, got {type(v).__name__}"
        raise TypeError(msg)
    if v.tzinfo is None:
        return v.replace(tzinfo=UTC)
    return v


class PointInTimeQuery:
    """All reads from this instance are filtered to ``as_of <= decision_time``.

    Pattern::

        # Backtest at session close of 2024-03-15 UTC
        q = PointInTimeQuery(store, as_of=datetime(2024, 3, 15, 21, 0, tzinfo=UTC))
        bars = q.equity_bars("NVDA", start=date(2024, 1, 1))
        # bars never contains any record with as_of > decision_time.

    Thread-safety: a fresh DuckDB connection is opened per query. Safe to
    share one ``PointInTimeQuery`` instance across threads.
    """

    def __init__(self, store: ParquetStore, *, as_of: datetime) -> None:
        if as_of.tzinfo is None:
            msg = "as_of must be timezone-aware (UTC)."
            raise ValueError(msg)
        self._store = store
        self._as_of = as_of.astimezone(UTC)

    @property
    def as_of(self) -> datetime:
        return self._as_of

    # ── equity ────────────────────────────────────────────────────────────

    def equity_bars(
        self,
        symbol: str,
        *,
        start: date | None = None,
        end: date | None = None,
        interval: str = "1d",
    ) -> list[EquityBar]:
        """Bars for ``symbol`` at ``interval``, optionally bounded by ``[start, end]``.

        Always filtered to ``as_of <= decision_time``. Returned in chronological order.
        """
        sym = symbol.upper()
        sym_root = partition.equity_symbol_root(self._store.base_dir, self._store.env, sym)
        if not sym_root.exists() or not any(sym_root.rglob("bars.parquet")):
            return []

        glob = _to_posix(sym_root / "**" / "bars.parquet")
        sql = (
            "SELECT * FROM read_parquet(?, hive_partitioning=1) "
            "WHERE as_of <= ? AND interval = ?"
        )
        params: list[Any] = [glob, self._as_of, interval]
        if start is not None:
            sql += " AND CAST(as_of AS DATE) >= ?"
            params.append(start)
        if end is not None:
            sql += " AND CAST(as_of AS DATE) <= ?"
            params.append(end)
        sql += " ORDER BY as_of"

        rows, cols = _execute(sql, params)
        return [_row_to_equity_bar(dict(zip(cols, r, strict=True))) for r in rows]

    def latest_quote(self, symbol: str, *, interval: str = "1d") -> EquityBar | None:
        """Most recent bar for ``symbol`` with ``as_of <= decision_time``."""
        sym = symbol.upper()
        sym_root = partition.equity_symbol_root(self._store.base_dir, self._store.env, sym)
        if not sym_root.exists() or not any(sym_root.rglob("bars.parquet")):
            return None
        glob = _to_posix(sym_root / "**" / "bars.parquet")
        sql = (
            "SELECT * FROM read_parquet(?, hive_partitioning=1) "
            "WHERE as_of <= ? AND interval = ? "
            "ORDER BY as_of DESC LIMIT 1"
        )
        rows, cols = _execute(sql, [glob, self._as_of, interval])
        if not rows:
            return None
        return _row_to_equity_bar(dict(zip(cols, rows[0], strict=True)))

    # ── options ───────────────────────────────────────────────────────────

    def options_chain_at(
        self, underlying: str, snapshot_date: date
    ) -> OptionsChainSnapshot | None:
        """Chain for ``underlying`` as captured on ``snapshot_date``.

        Returns ``None`` if no rows survive the ``as_of <= decision_time``
        filter — including the case where the snapshot was captured after
        the query's decision time (look-ahead is silently impossible).
        """
        und = underlying.upper()
        und_root = partition.options_underlying_root(self._store.base_dir, self._store.env, und)
        if not und_root.exists():
            return None

        pattern = f"expiry=*/snapshot={snapshot_date.isoformat()}.parquet"
        files = list(und_root.glob(pattern))
        if not files:
            return None

        glob = _to_posix(und_root / pattern)
        sql = (
            "SELECT * FROM read_parquet(?, hive_partitioning=1) "
            "WHERE as_of <= ? "
            "ORDER BY expiration, strike"
        )
        rows, cols = _execute(sql, [glob, self._as_of])
        if not rows:
            return None

        contracts = [_row_to_option_contract(dict(zip(cols, r, strict=True))) for r in rows]
        first_row = dict(zip(cols, rows[0], strict=True))
        return OptionsChainSnapshot(
            as_of=_ensure_utc(first_row["as_of"]),
            underlying=und,
            underlying_price=float(first_row["underlying_price"]),
            contracts=tuple(contracts),
            source=first_row["source"],
        )

    # ── corporate actions ─────────────────────────────────────────────────

    def corporate_actions(
        self, symbol: str, *, start: date | None = None
    ) -> list[CorporateAction]:
        sym = symbol.upper()
        path = partition.corporate_actions_file(self._store.base_dir, self._store.env, sym)
        if not path.exists():
            return []
        sql = "SELECT * FROM read_parquet(?) WHERE as_of <= ?"
        params: list[Any] = [_to_posix(path), self._as_of]
        if start is not None:
            sql += " AND ex_date >= ?"
            params.append(start)
        sql += " ORDER BY ex_date"
        rows, cols = _execute(sql, params)
        return [_row_to_corp_action(dict(zip(cols, r, strict=True))) for r in rows]

    # ── insider trades ────────────────────────────────────────────────────

    def insider_trades(
        self,
        ticker: str,
        *,
        start: date | None = None,
        trade_type_filter: str | None = None,
    ) -> list[InsiderTrade]:
        """Insider trades for ``ticker``, filtered to as_of <= decision_time.

        :param trade_type_filter: substring match on trade_type (case-insensitive).
            Use ``"purchase"`` to get only buys, ``"sale"`` for sells.
        """
        sym = ticker.upper()
        sym_root = partition.insider_trades_symbol_root(
            self._store.base_dir, self._store.env, sym
        )
        if not sym_root.exists() or not any(sym_root.rglob("trades.parquet")):
            return []
        glob = _to_posix(sym_root / "**" / "trades.parquet")
        sql = "SELECT * FROM read_parquet(?, hive_partitioning=1) WHERE as_of <= ?"
        params: list[Any] = [glob, self._as_of]
        if start is not None:
            sql += " AND filing_date >= ?"
            params.append(start)
        if trade_type_filter is not None:
            sql += " AND lower(trade_type) LIKE ?"
            params.append(f"%{trade_type_filter.lower()}%")
        sql += " ORDER BY filing_date DESC, trade_date DESC"
        rows, cols = _execute(sql, params)
        return [_row_to_insider_trade(dict(zip(cols, r, strict=True))) for r in rows]

    # ── earnings ──────────────────────────────────────────────────────────

    def earnings(
        self, symbol: str, *, start: date | None = None
    ) -> list[EarningsEvent]:
        sym = symbol.upper()
        path = partition.earnings_file(self._store.base_dir, self._store.env, sym)
        if not path.exists():
            return []
        sql = "SELECT * FROM read_parquet(?) WHERE as_of <= ?"
        params: list[Any] = [_to_posix(path), self._as_of]
        if start is not None:
            sql += " AND report_date >= ?"
            params.append(start)
        sql += " ORDER BY report_date"
        rows, cols = _execute(sql, params)
        return [_row_to_earnings(dict(zip(cols, r, strict=True))) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
#  DuckDB helper
# ─────────────────────────────────────────────────────────────────────────────


def _execute(sql: str, params: list[Any]) -> tuple[list[tuple[Any, ...]], list[str]]:
    """Run ``sql`` with bound ``params`` in a one-shot in-memory connection."""
    conn = duckdb.connect(":memory:")
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return rows, cols
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Row → pydantic adapters
# ─────────────────────────────────────────────────────────────────────────────


def _row_to_equity_bar(row: dict[str, Any]) -> EquityBar:
    return EquityBar(
        as_of=_ensure_utc(row["as_of"]),
        symbol=row["symbol"],
        interval=row["interval"],
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=int(row["volume"]),
        source=row["source"],
    )


def _row_to_option_contract(row: dict[str, Any]) -> OptionContract:
    return OptionContract(
        as_of=_ensure_utc(row["as_of"]),
        underlying=row["underlying"],
        expiration=row["expiration"],
        strike=float(row["strike"]),
        right=OptionRight(row["right"]),
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        last=row.get("last"),
        volume=int(row.get("volume") or 0),
        open_interest=int(row.get("open_interest") or 0),
        iv=row.get("iv"),
        delta=row.get("delta"),
        gamma=row.get("gamma"),
        theta=row.get("theta"),
        vega=row.get("vega"),
        rho=row.get("rho"),
        source=row["source"],
    )


def _row_to_insider_trade(row: dict[str, Any]) -> InsiderTrade:
    return InsiderTrade(
        as_of=_ensure_utc(row["as_of"]),
        filing_date=row["filing_date"],
        trade_date=row["trade_date"],
        ticker=row["ticker"],
        company=row.get("company"),
        insider_name=row["insider_name"],
        insider_title=row.get("insider_title"),
        trade_type=row["trade_type"],
        price_per_share=row.get("price_per_share"),
        quantity=int(row["quantity"]),
        shares_owned_after=(
            int(row["shares_owned_after"])
            if row.get("shares_owned_after") is not None
            else None
        ),
        dollar_value=row.get("dollar_value"),
        source=row["source"],
    )


def _row_to_corp_action(row: dict[str, Any]) -> CorporateAction:
    return CorporateAction(
        as_of=_ensure_utc(row["as_of"]),
        symbol=row["symbol"],
        ex_date=row["ex_date"],
        type=row["type"],
        ratio=row.get("ratio"),
        cash_amount=row.get("cash_amount"),
        source=row["source"],
    )


def _row_to_earnings(row: dict[str, Any]) -> EarningsEvent:
    return EarningsEvent(
        as_of=_ensure_utc(row["as_of"]),
        symbol=row["symbol"],
        report_date=row["report_date"],
        when=row["when"],
        eps_estimate=row.get("eps_estimate"),
        eps_actual=row.get("eps_actual"),
        eps_surprise_pct=row.get("eps_surprise_pct"),
        source=row["source"],
    )


# Helper that's also useful for pyarrow-side metadata reads (kept here so
# callers don't import pyarrow directly).
def options_snapshot_metadata(path: Path) -> dict[str, str]:
    """Read parquet-level key/value metadata from an options snapshot file."""
    schema = pq.read_schema(path)
    meta = schema.metadata or {}
    return {k.decode("utf-8"): v.decode("utf-8") for k, v in meta.items()}
