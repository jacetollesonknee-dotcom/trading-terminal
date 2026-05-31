"""OpenInsider client tests — respx-mocked HTML fixtures.

Cover parsing (good rows, missing optional cells, malformed rows),
error paths (network failure, empty page), and round-trip persistence
through ParquetStore + PointInTimeQuery.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from textwrap import dedent
from typing import Final

import httpx
import pytest
import respx

from ingestion._http import HTTPError
from ingestion.openinsider_client import (
    OpenInsiderClient,
    OpenInsiderError,
    _parse_trades,
)
from storage.query import PointInTimeQuery
from storage.store import ParquetStore

_BASE: Final[str] = "http://openinsider.com"


def _table(rows_html: str) -> str:
    """Wrap row HTML in a minimal valid OpenInsider table.tinytable shell."""
    return dedent(
        f"""\
        <html><body>
        <table class="tinytable">
          <thead><tr><th>x</th><th>Filing</th><th>Trade</th><th>Tkr</th>
            <th>Co</th><th>Insider</th><th>Title</th><th>Type</th>
            <th>Price</th><th>Qty</th><th>Owned</th><th>Value</th></tr></thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
        </body></html>
        """
    )


_ROW_GOOD = """
<tr>
  <td>+</td>
  <td>2026-05-30 16:23:11</td>
  <td>2026-05-29</td>
  <td>NVDA</td>
  <td>NVIDIA Corp</td>
  <td>Huang Jen-Hsun</td>
  <td>CEO</td>
  <td>P - Purchase</td>
  <td>$110.50</td>
  <td>10,000</td>
  <td>500,000</td>
  <td>$1,105,000</td>
</tr>"""

_ROW_BLANK_PRICE = """
<tr>
  <td>+</td>
  <td>2026-05-30</td>
  <td>2026-05-29</td>
  <td>NVDA</td>
  <td>NVIDIA Corp</td>
  <td>Holder Beneficial</td>
  <td>10% Owner</td>
  <td>S+OE</td>
  <td></td>
  <td>1,500</td>
  <td></td>
  <td></td>
</tr>"""

_ROW_MALFORMED = "<tr><td>+</td><td>not enough cells</td></tr>"


# ─────────────────────────────────────────────────────────────────────────
#  Parser direct tests
# ─────────────────────────────────────────────────────────────────────────


def test_parse_well_formed_row() -> None:
    html = _table(_ROW_GOOD)
    trades = _parse_trades(html, fallback_ticker=None, as_of=datetime.now(UTC))
    assert len(trades) == 1
    t = trades[0]
    assert t.ticker == "NVDA"
    assert t.insider_name == "Huang Jen-Hsun"
    assert t.insider_title == "CEO"
    assert t.trade_type == "P - Purchase"
    assert t.price_per_share == 110.50
    assert t.quantity == 10_000
    assert t.shares_owned_after == 500_000
    assert t.dollar_value == 1_105_000
    assert t.filing_date == date(2026, 5, 30)
    assert t.trade_date == date(2026, 5, 29)


def test_parse_handles_blank_optional_cells() -> None:
    trades = _parse_trades(
        _table(_ROW_BLANK_PRICE), fallback_ticker=None, as_of=datetime.now(UTC)
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.price_per_share is None
    assert t.shares_owned_after is None
    assert t.dollar_value is None
    assert t.quantity == 1500


def test_parser_skips_malformed_rows_silently() -> None:
    trades = _parse_trades(
        _table(_ROW_GOOD + _ROW_MALFORMED + _ROW_BLANK_PRICE),
        fallback_ticker=None,
        as_of=datetime.now(UTC),
    )
    assert len(trades) == 2


def test_parser_returns_empty_when_no_table() -> None:
    trades = _parse_trades("<html><body>no table</body></html>",
                           fallback_ticker=None, as_of=datetime.now(UTC))
    assert trades == []


def test_parser_uses_fallback_ticker_when_cell_empty() -> None:
    row_no_ticker = _ROW_GOOD.replace("<td>NVDA</td>", "<td></td>")
    trades = _parse_trades(
        _table(row_no_ticker), fallback_ticker="AAPL", as_of=datetime.now(UTC)
    )
    assert trades[0].ticker == "AAPL"


# ─────────────────────────────────────────────────────────────────────────
#  Client request paths
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_get_trades_for_symbol_round_trip() -> None:
    respx.get(f"{_BASE}/screener").mock(
        return_value=httpx.Response(200, text=_table(_ROW_GOOD))
    )
    with OpenInsiderClient() as oi:
        trades = oi.get_trades_for_symbol("NVDA")
    assert len(trades) == 1
    assert trades[0].ticker == "NVDA"


@respx.mock
def test_get_latest_trades_path() -> None:
    respx.get(f"{_BASE}/latest-insider-trading").mock(
        return_value=httpx.Response(200, text=_table(_ROW_GOOD))
    )
    with OpenInsiderClient() as oi:
        trades = oi.get_latest_trades()
    assert len(trades) == 1


@respx.mock
def test_get_top_purchases_of_month_path() -> None:
    respx.get(f"{_BASE}/top-insider-purchases-of-the-month").mock(
        return_value=httpx.Response(200, text=_table(_ROW_GOOD))
    )
    with OpenInsiderClient() as oi:
        trades = oi.get_top_purchases_of_month()
    assert len(trades) == 1


@respx.mock
def test_network_failure_wraps_in_openinsider_error() -> None:
    respx.get(f"{_BASE}/latest-insider-trading").mock(
        side_effect=httpx.TransportError("boom")
    )
    with OpenInsiderClient() as oi:  # noqa: SIM117
        with pytest.raises(OpenInsiderError, match="failed"):
            oi.get_latest_trades()


@respx.mock
def test_404_raises_openinsider_error() -> None:
    respx.get(f"{_BASE}/latest-insider-trading").mock(
        return_value=httpx.Response(404, text="not found")
    )
    with OpenInsiderClient() as oi:  # noqa: SIM117
        with pytest.raises(OpenInsiderError):
            oi.get_latest_trades()


def test_http_error_class_is_imported_for_clarity() -> None:
    """Smoke: ensure HTTPError is importable so callers can catch it
    alongside OpenInsiderError if they reuse the retry helper directly."""
    assert HTTPError is not None


# ─────────────────────────────────────────────────────────────────────────
#  Storage round trip
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_storage_round_trip(tmp_path: Path) -> None:
    respx.get(f"{_BASE}/screener").mock(
        return_value=httpx.Response(200, text=_table(_ROW_GOOD + _ROW_BLANK_PRICE))
    )
    store = ParquetStore(tmp_path, env="test")
    with OpenInsiderClient() as oi:
        trades = oi.get_trades_for_symbol("NVDA")
    result = store.write_insider_trades(trades)
    assert result.persisted == 2

    q = PointInTimeQuery(store, as_of=datetime.now(UTC))
    got = q.insider_trades("NVDA")
    assert len(got) == 2
    # Result sorted DESC by filing_date — same date here, both visible
    assert {t.insider_name for t in got} == {"Huang Jen-Hsun", "Holder Beneficial"}


@respx.mock
def test_dedup_across_polls(tmp_path: Path) -> None:
    """Re-polling the same page must not duplicate the same trade."""
    respx.get(f"{_BASE}/screener").mock(
        return_value=httpx.Response(200, text=_table(_ROW_GOOD))
    )
    store = ParquetStore(tmp_path, env="test")
    with OpenInsiderClient() as oi:
        t1 = oi.get_trades_for_symbol("NVDA")
        t2 = oi.get_trades_for_symbol("NVDA")
    r1 = store.write_insider_trades(t1)
    r2 = store.write_insider_trades(t2)
    assert r1.persisted == 1
    assert r2.persisted == 0
    assert r2.deduplicated == 1


@respx.mock
def test_filter_by_trade_type(tmp_path: Path) -> None:
    """`trade_type_filter='purchase'` returns only buys."""
    # The fixture has one purchase and one S+OE (sale-ish)
    respx.get(f"{_BASE}/screener").mock(
        return_value=httpx.Response(200, text=_table(_ROW_GOOD + _ROW_BLANK_PRICE))
    )
    store = ParquetStore(tmp_path, env="test")
    with OpenInsiderClient() as oi:
        store.write_insider_trades(oi.get_trades_for_symbol("NVDA"))

    q = PointInTimeQuery(store, as_of=datetime.now(UTC))
    purchases = q.insider_trades("NVDA", trade_type_filter="purchase")
    assert len(purchases) == 1
    assert "Purchase" in purchases[0].trade_type
