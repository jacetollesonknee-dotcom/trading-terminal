"""Error-path tests for YahooClient + the retry helper.

Covers:
* 429 → retry → success
* 5xx → retry → exhaustion → TransientHTTPError
* 4xx (non-429) → immediate HTTPError, no retry
* Transport-layer exceptions → retry → exhaustion → TransientHTTPError
* JSON parse failure → YahooClientError
* Yahoo error envelope → YahooClientError
* Retry-After header on 429 → honored
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from ingestion._http import (
    HTTPError,
    TransientHTTPError,
    _parse_retry_after,
    get_with_retry,
)
from ingestion.yahoo_client import YahooClient, YahooClientError

_URL_RE = "https://query1.finance.yahoo.com/v8/finance/chart/.*"


def _ok_payload() -> dict[str, Any]:
    return {
        "chart": {
            "result": [{
                "meta": {"symbol": "NVDA"},
                "timestamp": [],
                "indicators": {"quote": [{
                    "open": [], "high": [], "low": [], "close": [], "volume": [],
                }]},
            }],
            "error": None,
        }
    }


# ─────────────────────────────────────────────────────────────────────────
#  YahooClient — Yahoo-layer errors
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_yahoo_error_envelope_raises() -> None:
    payload: dict[str, Any] = {
        "chart": {
            "result": None,
            "error": {"code": "Not Found", "description": "Bad symbol"},
        },
    }
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))

    with YahooClient() as yc, pytest.raises(YahooClientError, match="Bad symbol"):
        yc.get_history("BADSYM")


@respx.mock
def test_non_json_response_raises_yahoo_error() -> None:
    respx.get(url__regex=_URL_RE).mock(
        return_value=httpx.Response(200, content=b"<html>nope</html>",
                                    headers={"Content-Type": "text/html"}),
    )
    with YahooClient() as yc, pytest.raises(YahooClientError, match="non-JSON"):
        yc.get_history("NVDA")


@respx.mock
def test_empty_result_array_raises_yahoo_error() -> None:
    payload: dict[str, Any] = {"chart": {"result": [], "error": None}}
    respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(200, json=payload))
    with YahooClient() as yc, pytest.raises(YahooClientError, match="no chart result"):
        yc.get_history("NVDA")


# ─────────────────────────────────────────────────────────────────────────
#  YahooClient — HTTP layer (via retry helper)
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_429_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Skip the sleeps so the test stays fast.
    monkeypatch.setattr("ingestion._http.time.sleep", lambda *_: None)

    route = respx.get(url__regex=_URL_RE)
    route.side_effect = [
        httpx.Response(429),
        httpx.Response(200, json=_ok_payload()),
    ]

    with YahooClient(retries=2) as yc:
        bars = yc.get_history("NVDA")
    assert bars == []
    assert route.call_count == 2


@respx.mock
def test_500_retried_then_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingestion._http.time.sleep", lambda *_: None)
    route = respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(500))

    with YahooClient(retries=2) as yc, pytest.raises(TransientHTTPError, match="3 attempts"):
        yc.get_history("NVDA")
    assert route.call_count == 3


@respx.mock
def test_404_raises_immediately_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingestion._http.time.sleep", lambda *_: None)
    route = respx.get(url__regex=_URL_RE).mock(return_value=httpx.Response(404))

    with YahooClient(retries=3) as yc, pytest.raises(HTTPError, match="status=404"):
        yc.get_history("NVDA")
    assert route.call_count == 1  # no retry on 4xx


@respx.mock
def test_transport_error_retried_then_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingestion._http.time.sleep", lambda *_: None)
    route = respx.get(url__regex=_URL_RE).mock(side_effect=httpx.ConnectError("boom"))

    with YahooClient(retries=2) as yc, pytest.raises(TransientHTTPError, match="3 attempts"):
        yc.get_history("NVDA")
    assert route.call_count == 3


# ─────────────────────────────────────────────────────────────────────────
#  Retry helper unit-level
# ─────────────────────────────────────────────────────────────────────────


def test_parse_retry_after_seconds_form() -> None:
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after("0") == 0.0


def test_parse_retry_after_returns_none_for_date_form() -> None:
    # HTTP-date form is too rare and the consequence of falling back to the
    # backoff curve is fine. Anything non-numeric → None.
    assert _parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert _parse_retry_after(None) is None


@respx.mock
def test_retry_after_header_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    monkeypatch.setattr("ingestion._http.time.sleep", calls.append)

    route = respx.get(url__regex=_URL_RE)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(200, json=_ok_payload()),
    ]
    with httpx.Client(base_url="https://query1.finance.yahoo.com") as client:
        get_with_retry(client, "/v8/finance/chart/NVDA", params={"range": "5d"},
                       retries=2, backoff_s=0.1, backoff_cap_s=10.0)

    # The single sleep should match Retry-After (3s), not the curve (0.1s base).
    assert calls == [3.0]
