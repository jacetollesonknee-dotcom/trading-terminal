"""Schwab client tests against mocked HTTP.

The chain fixture below is the shape asserted in ``parse_chain``. If the
first live pull disagrees with it, the fixture is what to update — and the
parser follows.
"""

from __future__ import annotations

import base64
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx

from config.secrets import SchwabAppCredentials, SchwabToken
from config.settings import Settings
from ingestion.schema import OptionRight
from ingestion.schwab_client import (
    SchwabAPIError,
    SchwabAuthError,
    SchwabClient,
    SchwabRateLimitError,
    build_authorize_url,
    exchange_authorization_code,
    parse_chain,
)

BASE = "https://api.schwabapi.com"
_AS_OF = datetime(2026, 3, 2, 21, 0, tzinfo=UTC)


def _creds() -> SchwabAppCredentials:
    return SchwabAppCredentials(app_key="KEY", app_secret="SECRET")


def _token(*, expired: bool = False) -> SchwabToken:
    delta = timedelta(minutes=-5) if expired else timedelta(minutes=25)
    return SchwabToken(
        access_token="ACCESS", refresh_token="REFRESH", token_type="Bearer",
        expires_at=datetime.now(UTC) + delta, scope="api",
    )


def _settings() -> Settings:
    return Settings(_env_file=None, schwab_env="production", schwab_rate_limit_rps=1000.0)


def _row(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "putCall": "CALL", "symbol": "QQQ   260320C00500000", "bid": 5.1, "ask": 5.2,
        "last": 5.15, "mark": 5.15, "totalVolume": 1234, "openInterest": 5000,
        "volatility": 18.5, "delta": 0.52, "gamma": 0.03, "theta": -0.1, "vega": 0.4,
        "rho": 0.05, "strikePrice": 500.0, "daysToExpiration": 18, "multiplier": 100.0,
    }
    base.update(over)
    return base


def _payload() -> dict[str, Any]:
    return {
        "symbol": "QQQ", "status": "SUCCESS", "underlyingPrice": 500.12,
        "underlying": {"symbol": "QQQ", "last": 500.12},
        "callExpDateMap": {
            "2026-03-20:18": {
                "500.0": [_row()],
                "505.0": [_row(strikePrice=505.0, bid=2.0, ask=2.1, delta=0.35)],
            }
        },
        "putExpDateMap": {
            "2026-03-20:18": {
                "495.0": [_row(putCall="PUT", strikePrice=495.0, bid=3.0, ask=3.1, delta=-0.4)],
            }
        },
    }


@pytest.fixture
def keychain(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake the keychain at the client's import points; record writes."""
    state: dict[str, Any] = {"token": _token(), "creds": _creds(), "stored": []}

    def store_token(token: SchwabToken, *, env: str) -> None:
        state["stored"].append(token)

    monkeypatch.setattr("ingestion.schwab_client.get_schwab_token", lambda *, env: state["token"])
    monkeypatch.setattr(
        "ingestion.schwab_client.get_schwab_app_credentials", lambda *, env: state["creds"]
    )
    monkeypatch.setattr("ingestion.schwab_client.set_schwab_token", store_token)
    return state


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_s: float) -> None:
        return None

    monkeypatch.setattr("ingestion.schwab_client.asyncio.sleep", instant)


# ─────────────────────────────────────────────────────────────────────────────
#  parse_chain
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_chain_reads_both_maps_and_units() -> None:
    snap, dropped = parse_chain(_payload(), "QQQ", as_of=_AS_OF)
    assert dropped == 0
    assert snap.underlying_price == 500.12
    assert len(snap.contracts) == 3
    call = next(c for c in snap.contracts if c.strike == 500.0)
    assert call.right is OptionRight.call
    assert call.expiration == date(2026, 3, 20)  # from the map key
    assert call.iv == pytest.approx(0.185)  # percent -> fraction
    assert call.open_interest == 5000
    assert call.gamma == pytest.approx(0.03)
    assert call.source == "schwab"
    put = next(c for c in snap.contracts if c.right is OptionRight.put)
    assert put.strike == 495.0
    assert put.delta == pytest.approx(-0.4)


def test_parse_chain_maps_unavailable_greeks_to_none() -> None:
    p = _payload()
    p["callExpDateMap"]["2026-03-20:18"]["500.0"][0].update(
        {"delta": -999.0, "gamma": -999.0, "volatility": -999.0, "theta": -999.0}
    )
    snap, _ = parse_chain(p, "QQQ", as_of=_AS_OF)
    c = next(c for c in snap.contracts if c.strike == 500.0)
    assert c.delta is None and c.gamma is None and c.iv is None and c.theta is None


def test_parse_chain_drops_bad_rows_and_counts_them() -> None:
    p = _payload()
    p["callExpDateMap"]["2026-03-20:18"]["500.0"][0].update({"bid": 6.0, "ask": 5.0})  # crossed
    snap, dropped = parse_chain(p, "QQQ", as_of=_AS_OF)
    assert dropped == 1
    assert len(snap.contracts) == 2


def test_parse_chain_strike_range_is_inclusive() -> None:
    snap, _ = parse_chain(_payload(), "QQQ", as_of=_AS_OF, strike_range=(495.0, 500.0))
    assert sorted(c.strike for c in snap.contracts) == [495.0, 500.0]


def test_parse_chain_rejects_bad_status_and_empty() -> None:
    bad = _payload()
    bad["status"] = "FAILED"
    with pytest.raises(SchwabAPIError, match="status"):
        parse_chain(bad, "QQQ", as_of=_AS_OF)
    empty = _payload()
    empty["callExpDateMap"] = {}
    empty["putExpDateMap"] = {}
    with pytest.raises(SchwabAPIError, match="no usable contracts"):
        parse_chain(empty, "QQQ", as_of=_AS_OF)


def test_parse_chain_requires_underlying_price() -> None:
    p = _payload()
    p["underlyingPrice"] = 0.0
    p["underlying"] = {}
    with pytest.raises(SchwabAPIError, match="underlying price"):
        parse_chain(p, "QQQ", as_of=_AS_OF)


# ─────────────────────────────────────────────────────────────────────────────
#  Client: auth
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock(base_url=BASE, assert_all_called=False)
async def test_enter_with_valid_token_does_not_refresh(
    respx_mock: respx.MockRouter, keychain: dict[str, Any]
) -> None:
    refresh = respx_mock.post("/v1/oauth/token")
    async with SchwabClient(_settings()):
        pass
    assert not refresh.called
    assert keychain["stored"] == []


@pytest.mark.asyncio
@respx.mock(base_url=BASE)
async def test_enter_with_expired_token_refreshes_and_persists(
    respx_mock: respx.MockRouter, keychain: dict[str, Any]
) -> None:
    keychain["token"] = _token(expired=True)
    refresh = respx_mock.post("/v1/oauth/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "NEW", "refresh_token": "NEWR", "token_type": "Bearer",
            "expires_in": 1800, "scope": "api",
        })
    )
    async with SchwabClient(_settings()):
        pass
    assert refresh.called
    req = refresh.calls[0].request
    expected = "Basic " + base64.b64encode(b"KEY:SECRET").decode()
    assert req.headers["Authorization"] == expected
    assert b"grant_type=refresh_token" in req.content
    assert b"refresh_token=REFRESH" in req.content
    assert len(keychain["stored"]) == 1
    assert keychain["stored"][0].access_token == "NEW"
    assert not keychain["stored"][0].is_expired()


@pytest.mark.asyncio
async def test_enter_without_token_is_a_clear_error(keychain: dict[str, Any]) -> None:
    keychain["token"] = None
    with pytest.raises(SchwabAuthError, match="no Schwab token"):
        async with SchwabClient(_settings()):
            pass


@pytest.mark.asyncio
@respx.mock(base_url=BASE)
async def test_refresh_without_app_credentials_is_a_clear_error(
    respx_mock: respx.MockRouter, keychain: dict[str, Any]
) -> None:
    keychain["token"] = _token(expired=True)
    keychain["creds"] = None
    with pytest.raises(SchwabAuthError, match="app credentials"):
        async with SchwabClient(_settings()):
            pass


@pytest.mark.asyncio
@respx.mock(base_url=BASE)
async def test_refresh_rejected_by_schwab(
    respx_mock: respx.MockRouter, keychain: dict[str, Any]
) -> None:
    keychain["token"] = _token(expired=True)
    respx_mock.post("/v1/oauth/token").mock(return_value=httpx.Response(400, text="invalid_grant"))
    with pytest.raises(SchwabAuthError, match="refresh failed"):
        async with SchwabClient(_settings()):
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Client: chains
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock(base_url=BASE)
async def test_get_options_chain_sends_bearer_and_params(
    respx_mock: respx.MockRouter, keychain: dict[str, Any]
) -> None:
    route = respx_mock.get("/marketdata/v1/chains").mock(
        return_value=httpx.Response(200, json=_payload())
    )
    async with SchwabClient(_settings()) as c:
        snap = await c.get_options_chain("qqq", expiration=date(2026, 3, 20))
    req = route.calls[0].request
    assert req.headers["Authorization"] == "Bearer ACCESS"
    assert req.url.params["symbol"] == "QQQ"
    assert req.url.params["contractType"] == "ALL"
    assert req.url.params["fromDate"] == "2026-03-20"
    assert req.url.params["toDate"] == "2026-03-20"
    assert snap.underlying == "QQQ"
    assert len(snap.contracts) == 3
    assert snap.as_of.tzinfo is not None


@pytest.mark.asyncio
@respx.mock(base_url=BASE)
async def test_401_triggers_one_refresh_then_retries(
    respx_mock: respx.MockRouter, keychain: dict[str, Any]
) -> None:
    respx_mock.post("/v1/oauth/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "NEW", "refresh_token": "R", "token_type": "Bearer", "expires_in": 1800,
        })
    )
    route = respx_mock.get("/marketdata/v1/chains").mock(
        side_effect=[httpx.Response(401), httpx.Response(200, json=_payload())]
    )
    async with SchwabClient(_settings()) as c:
        await c.get_options_chain("QQQ")
    assert route.call_count == 2
    assert route.calls[1].request.headers["Authorization"] == "Bearer NEW"


@pytest.mark.asyncio
@respx.mock(base_url=BASE)
async def test_429_backs_off_then_succeeds(
    respx_mock: respx.MockRouter, keychain: dict[str, Any]
) -> None:
    route = respx_mock.get("/marketdata/v1/chains").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(503),
            httpx.Response(200, json=_payload()),
        ]
    )
    async with SchwabClient(_settings()) as c:
        snap = await c.get_options_chain("QQQ")
    assert route.call_count == 3
    assert len(snap.contracts) == 3


@pytest.mark.asyncio
@respx.mock(base_url=BASE)
async def test_persistent_429_is_a_rate_limit_error(
    respx_mock: respx.MockRouter, keychain: dict[str, Any]
) -> None:
    respx_mock.get("/marketdata/v1/chains").mock(return_value=httpx.Response(429))
    with pytest.raises(SchwabRateLimitError):
        async with SchwabClient(_settings()) as c:
            await c.get_options_chain("QQQ")


@pytest.mark.asyncio
@respx.mock(base_url=BASE)
async def test_4xx_is_an_api_error(respx_mock: respx.MockRouter, keychain: dict[str, Any]) -> None:
    respx_mock.get("/marketdata/v1/chains").mock(return_value=httpx.Response(404, text="nope"))
    with pytest.raises(SchwabAPIError, match="status=404"):
        async with SchwabClient(_settings()) as c:
            await c.get_options_chain("QQQ")


@pytest.mark.asyncio
async def test_use_outside_context_is_an_error(keychain: dict[str, Any]) -> None:
    c = SchwabClient(_settings())
    with pytest.raises(SchwabAuthError, match="not authenticated"):
        await c.get_options_chain("QQQ")


# ─────────────────────────────────────────────────────────────────────────────
#  One-time authorization flow
# ─────────────────────────────────────────────────────────────────────────────


def test_build_authorize_url() -> None:
    url = build_authorize_url("KEY", "https://127.0.0.1:8182")
    assert url.startswith(f"{BASE}/v1/oauth/authorize?")
    assert "client_id=KEY" in url
    assert "redirect_uri=https%3A%2F%2F127.0.0.1%3A8182" in url


@respx.mock(base_url=BASE)
def test_exchange_authorization_code(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post("/v1/oauth/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "A", "refresh_token": "R", "token_type": "Bearer",
            "expires_in": 1800, "scope": "api",
        })
    )
    token = exchange_authorization_code(_creds(), "THECODE", "https://127.0.0.1:8182")
    assert token.access_token == "A"
    assert not token.is_expired()
    body = route.calls[0].request.content
    assert b"grant_type=authorization_code" in body
    assert b"code=THECODE" in body


@respx.mock(base_url=BASE)
def test_exchange_rejected(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("/v1/oauth/token").mock(return_value=httpx.Response(400, text="bad code"))
    with pytest.raises(SchwabAuthError, match="exchange failed"):
        exchange_authorization_code(_creds(), "X", "https://127.0.0.1:8182")
