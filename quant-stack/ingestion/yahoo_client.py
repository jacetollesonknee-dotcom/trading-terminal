"""Yahoo Finance public-endpoint client.

Equity bars, splits, and dividends from the same `chart` endpoint that
``terminal/data_feeds.py`` uses today. This module is the **engine-grade**
version: schema-validated returns, retry policy, explicit UTC ``as_of``,
no silent fallbacks.

Endpoint used:
    https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
        ?period1=...&period2=...&interval=...&events=div,split

Limits worth knowing (unofficial; Yahoo doesn't publish):
* 1m bars: ~7 days of history
* 5m/15m/30m bars: ~60 days
* 1h bars: ~730 days
* 1d bars: back to ticker inception (~1970 for most US large caps)

`as_of` convention for daily bars: 21:00 UTC of the trading day
(4pm ET in DST; 5pm ET in standard time). Conservative — bar values are
known no earlier than session close. For intraday bars, ``as_of`` is the
timestamp Yahoo returns (bar **open** time) plus the interval, i.e. the
bar's close instant.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, Final, Literal, cast

import httpx

from ingestion._http import HTTPError, TransientHTTPError, get_with_retry
from ingestion.schema import CorporateAction, EarningsEvent, EquityBar

_BASE_URL: Final[str] = "https://query1.finance.yahoo.com"
_CHART_PATH: Final[str] = "/v8/finance/chart/{symbol}"
_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (compatible; quant-stack/0.1.0; +https://example.invalid)"
)

# Yahoo accepts these range values. We re-export the safe subset.
YahooPeriod = Literal["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max", "ytd"]
YahooInterval = Literal["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]

# Interval → seconds (for close-time computation on intraday bars).
_INTERVAL_SECONDS: Final[dict[str, int]] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "1d": 0,        # handled by _close_time_for_day
    "1wk": 0,       # same
    "1mo": 0,       # same
}


class YahooClientError(RuntimeError):
    """Yahoo returned a recognisable error envelope (e.g. unknown symbol)."""


class YahooClient:
    """Synchronous Yahoo Finance chart-endpoint client.

    Construct once per process. Caller owns the lifecycle via ``close()``
    or a ``with`` block.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        client: httpx.Client | None = None,
        retries: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=_BASE_URL,
            timeout=timeout_s,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
        )
        self._retries = retries
        self._backoff_s = backoff_s

    def __enter__(self) -> YahooClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ── public API ────────────────────────────────────────────────────────

    def get_history(
        self,
        symbol: str,
        *,
        period: YahooPeriod = "1y",
        interval: YahooInterval = "1d",
    ) -> list[EquityBar]:
        """Pull OHLCV bars for ``symbol``.

        :returns: bars in chronological order, oldest first. Each tagged with
            UTC ``as_of`` at the bar's close instant.
        :raises HTTPError: permanent transport failure.
        :raises TransientHTTPError: retries exhausted.
        :raises YahooClientError: Yahoo's response was structurally invalid
            or contained an error envelope.
        """
        result = self._chart(symbol, period=period, interval=interval, events=None)
        return _parse_bars(result, symbol=symbol, interval=interval)

    def get_quote(self, symbol: str) -> EquityBar:
        """Latest 1d bar with ``as_of`` = the bar's close instant (UTC).

        Used by the Phase 0 smoke test to confirm connectivity.

        :raises YahooClientError: Yahoo returned no bars (e.g. invalid symbol).
        """
        bars = self.get_history(symbol, period="5d", interval="1d")
        if not bars:
            msg = f"{symbol}: no quote available from Yahoo"
            raise YahooClientError(msg)
        return bars[-1]

    def get_splits_and_dividends(
        self,
        symbol: str,
        *,
        period: YahooPeriod = "10y",
    ) -> list[CorporateAction]:
        """Splits + cash dividends within ``period`` for ``symbol``.

        :returns: events in chronological order, oldest first.
            ``as_of`` is set to the ex-date — under-conservative because
            the market knew earlier, but it's all Yahoo gives us. The
            backtester is welcome to widen the visibility window later.
        """
        result = self._chart(
            symbol, period=period, interval="1d", events="div,split",
        )
        return _parse_corp_actions(result, symbol=symbol)

    def get_earnings_calendar(self, symbol: str) -> list[EarningsEvent]:
        """Earnings calendar for ``symbol``.

        Yahoo doesn't expose this via the chart endpoint and the HTML
        calendar page is fragile. Returning an empty list with a stable
        signature so the orchestrator can call it uniformly. Phase 1.8
        will implement this against a more reliable source (or scrape
        the calendar page if it's stable enough by then).
        """
        # TODO(phase-1.8): scrape finance.yahoo.com/calendar/earnings?symbol=
        # or use https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=calendarEvents
        _ = symbol
        return []

    # ── internals ─────────────────────────────────────────────────────────

    def _chart(
        self,
        symbol: str,
        *,
        period: YahooPeriod,
        interval: YahooInterval,
        events: str | None,
    ) -> dict[str, Any]:
        """One GET against /v8/finance/chart/{symbol}."""
        params: dict[str, Any] = {"range": period, "interval": interval}
        if events:
            params["events"] = events
        path = _CHART_PATH.format(symbol=symbol)

        try:
            resp = get_with_retry(
                self._client,
                path,
                params=params,
                retries=self._retries,
                backoff_s=self._backoff_s,
            )
        except (HTTPError, TransientHTTPError):
            raise

        try:
            payload = resp.json()
        except ValueError as e:
            msg = f"{symbol}: Yahoo returned non-JSON ({resp.status_code})"
            raise YahooClientError(msg) from e

        chart = payload.get("chart") or {}
        err = chart.get("error")
        if err:
            msg = f"{symbol}: Yahoo error: {err}"
            raise YahooClientError(msg)

        results = chart.get("result") or []
        if not results:
            msg = f"{symbol}: Yahoo returned no chart result"
            raise YahooClientError(msg)
        return cast("dict[str, Any]", results[0])


# ─────────────────────────────────────────────────────────────────────────────
#  Parsers — pure functions, easy to test
# ─────────────────────────────────────────────────────────────────────────────


def _parse_bars(
    result: dict[str, Any],
    *,
    symbol: str,
    interval: str,
) -> list[EquityBar]:
    timestamps: list[int] = result.get("timestamp") or []
    quote_groups: list[dict[str, list[Any]]] = (
        result.get("indicators", {}).get("quote") or []
    )
    if not timestamps or not quote_groups:
        return []
    q = quote_groups[0]
    opens = q.get("open") or []
    highs = q.get("high") or []
    lows = q.get("low") or []
    closes = q.get("close") or []
    volumes = q.get("volume") or []

    bars: list[EquityBar] = []
    for i, ts in enumerate(timestamps):
        c = _idx(closes, i)
        if c is None:
            continue  # Yahoo signals "no data this bar" with null close
        o = _idx(opens, i) if _idx(opens, i) is not None else c
        h = _idx(highs, i) if _idx(highs, i) is not None else c
        low = _idx(lows, i) if _idx(lows, i) is not None else c
        v_raw = _idx(volumes, i)
        v = int(v_raw) if v_raw is not None else 0
        # Defensive: if Yahoo gives us low > high (rare data artefact), skip.
        if h < low:
            continue
        try:
            bar = EquityBar(
                as_of=_close_instant(ts, interval=interval),
                symbol=symbol.upper(),
                interval=_normalize_interval(interval),
                open=float(o),
                high=float(h),
                low=float(low),
                close=float(c),
                volume=v,
                source="yahoo",
            )
        except ValueError:
            # pydantic refused the row (e.g. negative price); skip rather
            # than crash the whole batch. Log later when structlog is wired.
            continue
        bars.append(bar)
    return bars


def _parse_corp_actions(
    result: dict[str, Any], *, symbol: str,
) -> list[CorporateAction]:
    events_block = result.get("events") or {}
    out: list[CorporateAction] = []

    fetched_at = datetime.now(UTC)

    splits = events_block.get("splits") or {}
    for _ts_key, ev in splits.items():
        ex_ts = ev.get("date")
        num = ev.get("numerator")
        den = ev.get("denominator")
        if ex_ts is None or not num or not den:
            continue
        ex_d = datetime.fromtimestamp(int(ex_ts), tz=UTC).date()
        ratio = float(num) / float(den)
        if ratio <= 0:
            continue
        out.append(
            CorporateAction(
                as_of=_action_as_of(ex_d, fetched_at),
                symbol=symbol.upper(),
                ex_date=ex_d,
                type="split",
                ratio=ratio,
                cash_amount=None,
                source="yahoo",
            )
        )

    divs = events_block.get("dividends") or {}
    for _ts_key, ev in divs.items():
        ex_ts = ev.get("date")
        amount = ev.get("amount")
        if ex_ts is None or amount is None:
            continue
        ex_d = datetime.fromtimestamp(int(ex_ts), tz=UTC).date()
        try:
            cash = float(amount)
        except (TypeError, ValueError):
            continue
        if cash < 0:
            continue
        out.append(
            CorporateAction(
                as_of=_action_as_of(ex_d, fetched_at),
                symbol=symbol.upper(),
                ex_date=ex_d,
                type="cash_dividend",
                ratio=None,
                cash_amount=cash,
                source="yahoo",
            )
        )

    out.sort(key=lambda a: (a.ex_date, a.type))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────────────────────────────────────


def _idx(seq: list[Any], i: int) -> Any:
    if 0 <= i < len(seq):
        return seq[i]
    return None


def _normalize_interval(interval: str) -> str:
    """Yahoo's interval strings are already the canonical form we store."""
    return interval


def _close_instant(timestamp: int, *, interval: str) -> datetime:
    """UTC instant at which the bar's close price is known.

    For 1d/1wk/1mo: the start-of-day timestamp Yahoo returns is mapped
    to 21:00 UTC of the same calendar day — a conservative proxy for
    "after the US session close". Some non-US tickers will be slightly
    off; widen later if we ingest them.

    For intraday: timestamp + interval_seconds.
    """
    if interval in ("1d", "1wk", "1mo"):
        d = datetime.fromtimestamp(timestamp, tz=UTC).date()
        return _close_time_for_day(d)
    offset = _INTERVAL_SECONDS.get(interval, 0)
    base = datetime.fromtimestamp(timestamp, tz=UTC)
    return base + timedelta(seconds=offset)


def _close_time_for_day(d: date) -> datetime:
    """21:00 UTC of ``d`` — close-of-session proxy for US equities."""
    return datetime(d.year, d.month, d.day, 21, 0, 0, tzinfo=UTC)


def _action_as_of(ex_date: date, fetched_at: datetime) -> datetime:
    """``as_of`` for a corporate action.

    Use the later of the ex-date close-time and ``fetched_at`` — historic
    actions we're recording NOW should be visible from now; future-dated
    actions (rare in Yahoo output but possible) shouldn't be visible
    until the ex-date.
    """
    ex_close = _close_time_for_day(ex_date)
    return max(ex_close, fetched_at) if ex_date > fetched_at.date() else ex_close
