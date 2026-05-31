"""Happy-path tests for YahooClient using respx (httpx mock).

These tests use entirely fabricated payloads that mirror the structure
Yahoo's /v8/finance/chart endpoint returns. No live network access.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx

from ingestion.schema import CorporateAction, EquityBar
from ingestion.yahoo_client import YahooClient, YahooClientError

_URL_RE = "https://query1.finance.yahoo.com/v8/finance/chart/.*"


# ─────────────────────────────────────────────────────────────────────────
#  Helpers — fabricate Yahoo responses
# ─────────────────────────────────────────────────────────────────────────


def _ts(*, year: int, month: int, day: int, hour: int = 14, minute: int = 30) -> int:
    """Build a UTC unix timestamp for an intraday/daily bar reference."""
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp())


def _chart_payload(
    *,
    timestamps: list[int],
    opens: list[float | None],
    highs: list[float | None],
    lows: list[float | None],
    closes: list[float | None],
    volumes: list[int | None],
    events: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "meta": {"symbol": "NVDA", "instrumentType": "EQUITY"},
        "timestamp": timestamps,
        "indicators": {
            "quote": [{
                "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": volumes,
            }],
        },
    }
    if events:
        result["events"] = events
    return {"chart": {"result": [result], "error": None}}


# ─────────────────────────────────────────────────────────────────────────
#  get_history
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_get_history_parses_bars() -> None:
    payload = _chart_payload(
        timestamps=[_ts(year=2024, month=1, day=2), _ts(year=2024, month=1, day=3)],
        opens=[100.0, 101.0],
        highs=[102.0, 103.5],
        lows=[99.5, 100.5],
        closes=[101.5, 103.0],
        volumes=[1_000_000, 1_500_000],
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc:
        bars = yc.get_history("NVDA", period="5d", interval="1d")

    assert len(bars) == 2
    assert all(isinstance(b, EquityBar) for b in bars)
    assert bars[0].symbol == "NVDA"
    assert bars[0].close == 101.5
    assert bars[0].source == "yahoo"
    # 1d bar as_of: 21:00 UTC of the trading day.
    assert bars[0].as_of == datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    assert bars[1].as_of == datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


@respx.mock
def test_get_history_skips_null_closes() -> None:
    payload = _chart_payload(
        timestamps=[_ts(year=2024, month=1, day=2), _ts(year=2024, month=1, day=3)],
        opens=[100.0, 101.0],
        highs=[102.0, 103.5],
        lows=[99.5, 100.5],
        closes=[101.5, None],  # Yahoo signals "no data" with null close
        volumes=[1_000_000, 1_500_000],
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc:
        bars = yc.get_history("NVDA", period="5d", interval="1d")

    assert len(bars) == 1
    assert bars[0].close == 101.5


@respx.mock
def test_get_history_intraday_as_of_at_close_instant() -> None:
    open_ts = _ts(year=2024, month=1, day=2, hour=14, minute=30)
    payload = _chart_payload(
        timestamps=[open_ts],
        opens=[100.0], highs=[100.5], lows=[99.9], closes=[100.2], volumes=[5000],
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc:
        bars = yc.get_history("NVDA", period="5d", interval="5m")

    # Intraday: as_of = open ts + 5 minutes.
    assert bars[0].as_of == datetime(2024, 1, 2, 14, 35, tzinfo=UTC)
    assert bars[0].interval == "5m"


@respx.mock
def test_get_history_empty_when_no_timestamps() -> None:
    payload = _chart_payload(
        timestamps=[],
        opens=[], highs=[], lows=[], closes=[], volumes=[],
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc:
        assert yc.get_history("XYZ") == []


@respx.mock
def test_get_history_drops_bar_when_high_less_than_low() -> None:
    """Yahoo occasionally returns inconsistent OHLC on illiquid bars. Skip not crash."""
    payload = _chart_payload(
        timestamps=[_ts(year=2024, month=1, day=2)],
        opens=[100.0], highs=[99.0], lows=[101.0],  # high<low
        closes=[100.0], volumes=[10],
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc:
        assert yc.get_history("NVDA") == []


# ─────────────────────────────────────────────────────────────────────────
#  get_quote
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_get_quote_returns_last_bar() -> None:
    payload = _chart_payload(
        timestamps=[_ts(year=2024, month=1, day=2), _ts(year=2024, month=1, day=3)],
        opens=[100.0, 101.0], highs=[102.0, 103.5], lows=[99.5, 100.5],
        closes=[101.5, 103.0], volumes=[1_000_000, 1_500_000],
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc:
        q = yc.get_quote("NVDA")

    assert q.close == 103.0
    assert q.as_of == datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


@respx.mock
def test_get_quote_raises_when_no_bars() -> None:
    payload = _chart_payload(
        timestamps=[], opens=[], highs=[], lows=[], closes=[], volumes=[],
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc, pytest.raises(YahooClientError, match="no quote"):
        yc.get_quote("BADSYM")


# ─────────────────────────────────────────────────────────────────────────
#  Splits + dividends
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_get_splits_and_dividends_parses_both() -> None:
    split_ts = _ts(year=2024, month=6, day=10)
    div_ts = _ts(year=2024, month=3, day=12)
    payload = _chart_payload(
        timestamps=[_ts(year=2024, month=1, day=2)],
        opens=[100.0], highs=[101.0], lows=[99.0], closes=[100.5], volumes=[1000],
        events={
            "splits": {
                str(split_ts): {"date": split_ts, "numerator": 10, "denominator": 1,
                                "splitRatio": "10:1"},
            },
            "dividends": {
                str(div_ts): {"date": div_ts, "amount": 0.04},
            },
        },
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc:
        actions = yc.get_splits_and_dividends("NVDA")

    assert len(actions) == 2
    assert all(isinstance(a, CorporateAction) for a in actions)

    by_type = {a.type: a for a in actions}
    assert by_type["split"].ratio == 10.0
    assert by_type["split"].cash_amount is None
    assert by_type["cash_dividend"].cash_amount == 0.04
    assert by_type["cash_dividend"].ratio is None


@respx.mock
def test_corp_actions_empty_when_no_events_block() -> None:
    payload = _chart_payload(
        timestamps=[_ts(year=2024, month=1, day=2)],
        opens=[100.0], highs=[101.0], lows=[99.0], closes=[100.5], volumes=[1000],
        events=None,
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc:
        assert yc.get_splits_and_dividends("NVDA") == []


@respx.mock
def test_corp_actions_skips_malformed_split() -> None:
    split_ts = _ts(year=2024, month=6, day=10)
    payload = _chart_payload(
        timestamps=[_ts(year=2024, month=1, day=2)],
        opens=[100.0], highs=[101.0], lows=[99.0], closes=[100.5], volumes=[1000],
        events={
            "splits": {
                str(split_ts): {"date": split_ts, "numerator": 0, "denominator": 1},
            },
        },
    )
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc:
        assert yc.get_splits_and_dividends("NVDA") == []


# ─────────────────────────────────────────────────────────────────────────
#  Earnings stub
# ─────────────────────────────────────────────────────────────────────────


def test_earnings_calendar_returns_empty_stub() -> None:
    """Placeholder until Phase 1.8 wires a real source."""
    with YahooClient() as yc:
        assert yc.get_earnings_calendar("NVDA") == []
