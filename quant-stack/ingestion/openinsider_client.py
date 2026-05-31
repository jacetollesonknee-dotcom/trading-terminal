"""OpenInsider client — Form-4 insider trades.

Public HTML at http://openinsider.com. No auth, no key. Polite polling
recommended — they update from SEC filings, not minute-by-minute, so 5
minutes during market hours is plenty.

Three pulls supported:

- ``get_trades_for_symbol(ticker)`` — recent trades for one ticker.
- ``get_latest_trades()`` — site-wide latest filings (any ticker).
- ``get_top_purchases_of_month()`` — cluster-buy leaderboard.

All three return ``list[InsiderTrade]`` with schema-validated records;
caller persists via ``ParquetStore.write_insider_trades`` which dedups
across overlapping fetches.

OpenInsider's HTML can drift; this parser is defensive about cell counts
and silently skips malformed rows (logs in a TODO). Per the brief's
"fails loudly" principle (#9), schema-validated *records* never silently
go through — a row that *does* parse must conform.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from types import TracebackType
from typing import Final

import httpx
from bs4 import BeautifulSoup, Tag

from ingestion._http import HTTPError, get_with_retry
from ingestion.schema import InsiderTrade

_BASE_URL: Final[str] = "http://openinsider.com"
_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (compatible; quant-stack/1.0; +https://github.com/)"
)
_DEFAULT_TIMEOUT_S: Final[float] = 10.0

# Cell positions in the standard OpenInsider table (table.tinytable).
# Index 0 is the X column we ignore; 1..11 are the data.
_CELL_FILING_DATE: Final[int] = 1
_CELL_TRADE_DATE: Final[int] = 2
_CELL_TICKER: Final[int] = 3
_CELL_COMPANY: Final[int] = 4
_CELL_INSIDER: Final[int] = 5
_CELL_TITLE: Final[int] = 6
_CELL_TRADE_TYPE: Final[int] = 7
_CELL_PRICE: Final[int] = 8
_CELL_QTY: Final[int] = 9
_CELL_OWNED: Final[int] = 10
_CELL_VALUE: Final[int] = 11
_MIN_CELLS: Final[int] = 12

_MONEY_RE: Final[re.Pattern[str]] = re.compile(r"[\$,+]")


class OpenInsiderError(RuntimeError):
    """OpenInsider request failed or returned an unparseable response."""


# ─────────────────────────────────────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────────────────────────────────────


class OpenInsiderClient:
    """Sync httpx client for openinsider.com.

    Used as a context manager so the underlying connection pool gets cleaned
    up deterministically.
    """

    def __init__(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self._client = httpx.Client(
            base_url=_BASE_URL,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
            timeout=timeout_s,
            follow_redirects=True,
        )

    def __enter__(self) -> OpenInsiderClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._client.close()

    # ── public API ────────────────────────────────────────────────────────

    def get_trades_for_symbol(self, ticker: str, *, lookback_days: int = 30) -> list[InsiderTrade]:
        """Recent insider trades for ``ticker`` within ``lookback_days``."""
        ticker = ticker.upper()
        params = {
            "s": ticker,
            "fd": str(lookback_days),  # filing date lookback
            "cnt": "100",              # row count
            "page": "1",
        }
        html = self._get("/screener", params=params)
        as_of = datetime.now(UTC)
        return _parse_trades(html, fallback_ticker=ticker, as_of=as_of)

    def get_latest_trades(self) -> list[InsiderTrade]:
        """Site-wide latest filings — any ticker. Polled for the dashboard."""
        html = self._get("/latest-insider-trading")
        as_of = datetime.now(UTC)
        return _parse_trades(html, fallback_ticker=None, as_of=as_of)

    def get_top_purchases_of_month(self) -> list[InsiderTrade]:
        """Cluster-buy leaderboard for the month — strong sentiment signal."""
        html = self._get("/top-insider-purchases-of-the-month")
        as_of = datetime.now(UTC)
        return _parse_trades(html, fallback_ticker=None, as_of=as_of)

    # ── internals ─────────────────────────────────────────────────────────

    def _get(self, path: str, *, params: dict[str, str] | None = None) -> str:
        try:
            resp = get_with_retry(self._client, path, params=params or {})
        except HTTPError as e:
            msg = f"OpenInsider GET {path} failed: {e}"
            raise OpenInsiderError(msg) from e
        return resp.text


# ─────────────────────────────────────────────────────────────────────────────
#  Parser — pure function, exported for direct testing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_trades(
    html: str,
    *,
    fallback_ticker: str | None,
    as_of: datetime,
) -> list[InsiderTrade]:
    """Parse an OpenInsider table.tinytable into InsiderTrade records.

    Rows that fail to parse are silently skipped (the HTML can drift).
    Rows that DO parse are then schema-validated — pydantic rejects any
    that fail to conform.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.tinytable")
    if not table:
        return []

    trades: list[InsiderTrade] = []
    tbody = table.find("tbody")
    if not isinstance(tbody, Tag):
        return []
    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < _MIN_CELLS:
            continue
        try:
            trades.append(_row_to_trade(cells, fallback_ticker=fallback_ticker, as_of=as_of))
        except (ValueError, IndexError, AttributeError):
            # Bad row; drop silently. Logging hook lands in scheduler.
            continue
    return trades


def _row_to_trade(
    cells: list[Tag],
    *,
    fallback_ticker: str | None,
    as_of: datetime,
) -> InsiderTrade:
    """Parse one <tr>'s cells into a validated InsiderTrade."""
    ticker = _text(cells[_CELL_TICKER]) or fallback_ticker or ""
    return InsiderTrade(
        as_of=as_of,
        filing_date=_parse_date(_text(cells[_CELL_FILING_DATE])),
        trade_date=_parse_date(_text(cells[_CELL_TRADE_DATE])),
        ticker=ticker,
        company=_text(cells[_CELL_COMPANY]) or None,
        insider_name=_text(cells[_CELL_INSIDER]),
        insider_title=_text(cells[_CELL_TITLE]) or None,
        trade_type=_text(cells[_CELL_TRADE_TYPE]),
        price_per_share=_parse_money(_text(cells[_CELL_PRICE])),
        quantity=_parse_int(_text(cells[_CELL_QTY])),
        shares_owned_after=_parse_int_opt(_text(cells[_CELL_OWNED])),
        dollar_value=_parse_money(_text(cells[_CELL_VALUE])),
        source="openinsider",
    )


# ── tiny parse helpers ─────────────────────────────────────────────────────

def _text(cell: Tag) -> str:
    return cell.get_text(strip=True)


def _parse_date(s: str) -> date:
    """OpenInsider uses YYYY-MM-DD throughout."""
    # Filing-date cell may have a time suffix "2026-05-31 16:23:11"; we
    # only want the date.
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def _parse_int(s: str) -> int:
    if not s:
        return 0
    return int(_MONEY_RE.sub("", s))


def _parse_int_opt(s: str) -> int | None:
    if not s:
        return None
    return _parse_int(s)


def _parse_money(s: str) -> float | None:
    """OpenInsider prices and dollar values come like '$1,234.56' or '$1.2M'.

    For per-share price we never see suffixed magnitudes, so just strip $ and ,
    and parse a float. Empty cells return None.
    """
    if not s:
        return None
    cleaned = _MONEY_RE.sub("", s)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None
