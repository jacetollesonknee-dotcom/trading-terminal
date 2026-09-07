"""Schwab Trader API client.

Implemented here: the OAuth token lifecycle (load from keychain, refresh,
persist) and the options-chain pull that the capture-forward recorder
needs. Everything else in the :class:`~ingestion.brokers.base.BrokerClient`
surface (history, quotes, accounts, orders) still raises
``NotImplementedError`` with a phase pointer, so callers typecheck against
the full interface but can't accidentally route an order.

Endpoints (public docs at developer.schwab.com):

    POST /v1/oauth/token           refresh_token / authorization_code grants,
                                   Basic auth with app key:secret
    GET  /v1/oauth/authorize       browser login; redirects with ?code=
    GET  /marketdata/v1/chains     option chain, Greeks + OI included

Token storage lives in :mod:`config.secrets` (OS keychain). This module
neither reads nor writes ``.env`` files. The client is intentionally
separate from any Schwab integration in Cowork — two OAuth surfaces, two
tokens, isolated blast radius (ADR-002).

What is asserted about Schwab's response shape is written down in
:func:`parse_chain` and pinned by tests against a fixture; the first live
pull is where any divergence shows up, and it should be run by the
operator on their own machine with their own credentials.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
from datetime import UTC, date, datetime
from typing import Any, Final, Literal
from urllib.parse import urlencode

import httpx
from pydantic import ValidationError

from config.secrets import (
    SchwabAppCredentials,
    SchwabToken,
    get_schwab_app_credentials,
    get_schwab_token,
    set_schwab_token,
)
from config.settings import SchwabEnv, Settings
from ingestion.schema import (
    AccountSnapshot,
    EquityBar,
    OptionContract,
    OptionRight,
    OptionsChainSnapshot,
    Order,
    Position,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

_BASE_URLS: dict[SchwabEnv, str] = {
    SchwabEnv.production: "https://api.schwabapi.com",
    SchwabEnv.sandbox: "https://api-sandbox.schwabapi.com",  # placeholder; verify
}

_TOKEN_PATH: Final[str] = "/v1/oauth/token"  # noqa: S105 — a URL path, not a secret
_AUTHORIZE_PATH: Final[str] = "/v1/oauth/authorize"
_CHAINS_PATH: Final[str] = "/marketdata/v1/chains"

_RETRYABLE: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES: Final[int] = 3
_BACKOFF_BASE_S: Final[float] = 0.5
_BACKOFF_CAP_S: Final[float] = 8.0
_HTTP_UNAUTHORIZED: Final[int] = 401
_HTTP_TOO_MANY: Final[int] = 429
_HTTP_BAD_REQUEST: Final[int] = 400

# Schwab (inherited from TD) reports an unavailable greek / vol as -999.
_UNAVAILABLE: Final[float] = -999.0
_PCT: Final[float] = 100.0


class SchwabAPIError(RuntimeError):
    """Raised when the Schwab API returns a non-recoverable error."""


class SchwabAuthError(SchwabAPIError):
    """OAuth failed or token cannot be refreshed."""


class SchwabRateLimitError(SchwabAPIError):
    """429 from Schwab. Client backs off and retries; surfaces if persistent."""


# ─────────────────────────────────────────────────────────────────────────────
#  OAuth helpers (sync; used by the CLI's one-time connect flow)
# ─────────────────────────────────────────────────────────────────────────────


def _basic_auth(creds: SchwabAppCredentials) -> str:
    raw = f"{creds.app_key}:{creds.app_secret}".encode()
    return "Basic " + base64.b64encode(raw).decode("ascii")


def build_authorize_url(
    app_key: str, redirect_uri: str, env: SchwabEnv = SchwabEnv.production
) -> str:
    """The URL the operator opens in a browser to approve the app."""
    query = urlencode({"client_id": app_key, "redirect_uri": redirect_uri})
    return f"{_BASE_URLS[env]}{_AUTHORIZE_PATH}?{query}"


def exchange_authorization_code(
    creds: SchwabAppCredentials,
    code: str,
    redirect_uri: str,
    *,
    env: SchwabEnv = SchwabEnv.production,
    timeout_s: float = 15.0,
) -> SchwabToken:
    """Trade the ``?code=`` from the redirect for a token. One-time, sync.

    Raises:
        SchwabAuthError: Schwab rejected the code.
    """
    issued_at = datetime.now(UTC)
    resp = httpx.post(
        f"{_BASE_URLS[env]}{_TOKEN_PATH}",
        headers={
            "Authorization": _basic_auth(creds),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri},
        timeout=timeout_s,
    )
    if resp.status_code >= _HTTP_BAD_REQUEST:
        msg = (
            f"authorization code exchange failed: status={resp.status_code} "
            f"body={resp.text[:200]}"
        )
        raise SchwabAuthError(msg)
    return SchwabToken.from_oauth_response(resp.json(), issued_at=issued_at)


# ─────────────────────────────────────────────────────────────────────────────
#  Chain parsing (pure; pinned by tests)
# ─────────────────────────────────────────────────────────────────────────────


def _greek(value: object) -> float | None:
    """A greek / vol field, or None when Schwab marks it unavailable."""
    if value is None:
        return None
    v = float(str(value))
    return None if v == _UNAVAILABLE else v


def _underlying_price(payload: dict[str, Any]) -> float:
    for candidate in (
        payload.get("underlyingPrice"),
        (payload.get("underlying") or {}).get("last"),
        (payload.get("underlying") or {}).get("mark"),
    ):
        if candidate is not None and float(candidate) > 0.0:
            return float(candidate)
    msg = "chain payload has no positive underlying price"
    raise SchwabAPIError(msg)


def _contract(
    raw: dict[str, Any],
    *,
    underlying: str,
    expiration: date,
    strike_key: str,
    right: OptionRight,
    as_of: datetime,
) -> OptionContract:
    """One contract from Schwab's dict. Raises ValidationError on a bad row."""
    iv = _greek(raw.get("volatility"))
    delta = _greek(raw.get("delta"))
    gamma = _greek(raw.get("gamma"))
    last = raw.get("last")
    return OptionContract(
        as_of=as_of,
        underlying=underlying,
        expiration=expiration,
        strike=float(raw.get("strikePrice", strike_key)),
        right=right,
        bid=max(float(raw.get("bid") or 0.0), 0.0),
        ask=max(float(raw.get("ask") or 0.0), 0.0),
        last=(float(last) if last is not None and float(last) > 0.0 else None),
        volume=int(raw.get("totalVolume") or 0),
        open_interest=int(raw.get("openInterest") or 0),
        iv=(iv / _PCT if iv is not None and iv >= 0.0 else None),
        delta=(max(-1.0, min(1.0, delta)) if delta is not None else None),
        gamma=(gamma if gamma is not None and gamma >= 0.0 else None),
        theta=_greek(raw.get("theta")),
        vega=_greek(raw.get("vega")),
        rho=_greek(raw.get("rho")),
        source="schwab",
    )


def parse_chain(
    payload: dict[str, Any],
    underlying: str,
    *,
    as_of: datetime,
    strike_range: tuple[float, float] | None = None,
) -> tuple[OptionsChainSnapshot, int]:
    """Turn a ``/marketdata/v1/chains`` payload into a snapshot.

    Schwab groups contracts as ``callExpDateMap`` / ``putExpDateMap`` ->
    ``"YYYY-MM-DD:DTE"`` -> ``"strike"`` -> ``[contract, ...]``. The
    expiration is taken from the map key (stable) rather than the
    per-contract ``expirationDate`` (format has varied). ``volatility`` is
    in percent and is stored as a fraction; ``-999`` greeks become ``None``.

    Rows that fail schema validation (a crossed quote, a negative strike)
    are dropped and counted rather than failing the whole chain — but an
    empty result is an error, not an empty snapshot.

    Returns:
        ``(snapshot, n_dropped)``.
    """
    status = payload.get("status")
    if status is not None and str(status).upper() != "SUCCESS":
        msg = f"chain status={status!r} for {underlying}"
        raise SchwabAPIError(msg)
    spot = _underlying_price(payload)

    contracts: list[OptionContract] = []
    dropped = 0
    maps = (("callExpDateMap", OptionRight.call), ("putExpDateMap", OptionRight.put))
    for map_key, right in maps:
        for exp_key, by_strike in (payload.get(map_key) or {}).items():
            expiration = date.fromisoformat(str(exp_key).split(":", 1)[0])
            for strike_key, rows in (by_strike or {}).items():
                for raw in rows or []:
                    try:
                        c = _contract(
                            raw, underlying=underlying, expiration=expiration,
                            strike_key=str(strike_key), right=right, as_of=as_of,
                        )
                    except (ValidationError, ValueError, TypeError):
                        dropped += 1
                        continue
                    if strike_range is not None and not (
                        strike_range[0] <= c.strike <= strike_range[1]
                    ):
                        continue
                    contracts.append(c)
    if not contracts:
        msg = f"chain for {underlying} contained no usable contracts ({dropped} dropped)"
        raise SchwabAPIError(msg)
    snapshot = OptionsChainSnapshot(
        as_of=as_of,
        underlying=underlying,
        underlying_price=spot,
        contracts=tuple(contracts),
        source="schwab",
    )
    return snapshot, dropped


# ─────────────────────────────────────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────────────────────────────────────


class SchwabClient:
    """Async HTTP client for the Schwab Trader API.

    Lifecycle::

        async with SchwabClient(settings) as client:
            chain = await client.get_options_chain("QQQ")

    Entering the context loads the token from the keychain and refreshes it
    if it is within 60 s of expiry. A 401 mid-session triggers one refresh
    and one retry. 429 and 5xx back off exponentially up to three retries;
    persistent 429 raises :class:`SchwabRateLimitError`. Outbound requests
    are throttled to ``settings.schwab_rate_limit_rps``.
    """

    name = "schwab"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._env = settings.schwab_env
        self._base_url = _BASE_URLS[self._env]
        self._token: SchwabToken | None = None
        self._http: httpx.AsyncClient | None = None
        self._min_interval_s = 1.0 / settings.schwab_rate_limit_rps
        self._last_request_at = 0.0

    # ── async context manager ──────────────────────────────────────────

    async def __aenter__(self) -> SchwabClient:
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._settings.schwab_request_timeout_s,
            headers={"Accept": "application/json"},
        )
        try:
            await self.authenticate()
        except Exception:
            await self._http.aclose()
            self._http = None
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ── auth ───────────────────────────────────────────────────────────

    async def authenticate(self) -> None:
        """Load the token from the keychain; refresh if within 60 s of expiry.

        Raises:
            SchwabAuthError: No token seeded, or refresh failed.
        """
        token = get_schwab_token(env=self._env.value)
        if token is None:
            msg = (
                f"no Schwab token in keychain for env={self._env.value!r}. Run "
                f"`python -m cli connect schwab` or `python -m config.secrets "
                f"--import-token-file <path>`."
            )
            raise SchwabAuthError(msg)
        self._token = token
        if token.is_expired():
            await self.refresh_token()

    async def refresh_token(self) -> None:
        """Exchange the refresh token for a new access token; persist it.

        Raises:
            SchwabAuthError: App credentials missing, or Schwab rejected the refresh.
        """
        if self._token is None:
            msg = "refresh_token called before authenticate"
            raise SchwabAuthError(msg)
        creds = get_schwab_app_credentials(env=self._env.value)
        if creds is None:
            msg = (
                f"no Schwab app credentials in keychain for env={self._env.value!r}; "
                f"run `python -m config.secrets --set-schwab-app`."
            )
            raise SchwabAuthError(msg)
        issued_at = datetime.now(UTC)
        resp = await self._client().post(
            _TOKEN_PATH,
            headers={
                "Authorization": _basic_auth(creds),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "refresh_token", "refresh_token": self._token.refresh_token},
        )
        if resp.status_code >= _HTTP_BAD_REQUEST:
            msg = (
                f"token refresh failed: status={resp.status_code} body={resp.text[:200]}. "
                f"If the refresh token has expired (7 days), re-run `python -m cli connect schwab`."
            )
            raise SchwabAuthError(msg)
        new = SchwabToken.from_oauth_response(resp.json(), issued_at=issued_at)
        set_schwab_token(new, env=self._env.value)
        self._token = new

    # ── transport ──────────────────────────────────────────────────────

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            msg = "SchwabClient must be used as `async with`"
            raise RuntimeError(msg)
        return self._http

    async def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._min_interval_s - (now - self._last_request_at)
        if wait > 0.0:
            await asyncio.sleep(wait)
        self._last_request_at = time.monotonic()

    async def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with Bearer auth, one 401-refresh, and 429/5xx backoff."""
        if self._token is None:
            msg = "not authenticated"
            raise SchwabAuthError(msg)
        refreshed = False
        for attempt in range(_MAX_RETRIES + 1):
            await self._throttle()
            resp = await self._client().get(
                path, params=params,
                headers={"Authorization": f"Bearer {self._token.access_token}"},
            )
            if resp.status_code < _HTTP_BAD_REQUEST:
                body = resp.json()
                if not isinstance(body, dict):
                    msg = f"GET {path}: expected a JSON object"
                    raise SchwabAPIError(msg)
                return body
            if resp.status_code == _HTTP_UNAUTHORIZED and not refreshed:
                refreshed = True
                await self.refresh_token()
                continue
            if resp.status_code in _RETRYABLE and attempt < _MAX_RETRIES:
                retry_after = resp.headers.get("Retry-After")
                delay = min(_BACKOFF_BASE_S * (2**attempt), _BACKOFF_CAP_S)
                if retry_after is not None:
                    # Seconds-form only; an HTTP-date Retry-After falls back to the curve.
                    with contextlib.suppress(ValueError):
                        delay = min(float(retry_after), _BACKOFF_CAP_S)
                await asyncio.sleep(delay)
                continue
            if resp.status_code == _HTTP_TOO_MANY:
                msg = f"GET {path}: rate limited after {attempt + 1} attempts"
                raise SchwabRateLimitError(msg)
            msg = f"GET {path}: status={resp.status_code} body={resp.text[:200]}"
            raise SchwabAPIError(msg)
        msg = f"GET {path}: exhausted retries"
        raise SchwabAPIError(msg)

    # ── market data ────────────────────────────────────────────────────

    async def get_history(
        self,
        symbol: str,
        *,
        period: Literal["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd"],
        interval: Literal["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"] = "1d",
    ) -> list[EquityBar]:
        """Historical OHLCV bars. Not implemented — use the Yahoo client for history."""
        raise NotImplementedError("Phase 1.2 — use ingestion.yahoo_client for history.")

    async def get_quote(self, symbol: str) -> EquityBar:
        """Latest quote. Not implemented — use the Yahoo client."""
        raise NotImplementedError("Phase 1.2 — use ingestion.yahoo_client for quotes.")

    async def get_options_chain(
        self,
        underlying: str,
        *,
        expiration: date | None = None,
        strike_range: tuple[float, float] | None = None,
    ) -> OptionsChainSnapshot:
        """Pull the current chain for ``underlying`` with Greeks, IV, OI.

        Args:
            underlying: Ticker, uppercased.
            expiration: If given, only that expiry (``fromDate``/``toDate``).
            strike_range: ``(low, high)`` inclusive, applied client-side —
                the endpoint has no strike-range filter.

        Raises:
            SchwabAPIError: Bad status, no usable contracts, or HTTP failure.
        """
        params: dict[str, Any] = {
            "symbol": underlying.upper(),
            "contractType": "ALL",
            "includeUnderlyingQuote": "true",
        }
        if expiration is not None:
            params["fromDate"] = expiration.isoformat()
            params["toDate"] = expiration.isoformat()
        payload = await self._get_json(_CHAINS_PATH, params)
        snapshot, _dropped = parse_chain(
            payload, underlying.upper(), as_of=datetime.now(UTC), strike_range=strike_range
        )
        return snapshot

    # ── account / positions / orders (Phase 5+) ────────────────────────

    async def get_account(self, account_id: str) -> AccountSnapshot:
        raise NotImplementedError("Phase 5 — account state.")

    async def get_positions(self, account_id: str) -> list[Position]:
        raise NotImplementedError("Phase 5 — positions.")

    async def place_order(self, order: Order) -> Order:
        """Never routes without Settings.live_orders_armed AND human confirmation."""
        raise NotImplementedError("Phase 6/7 — order routing is gated.")

    async def cancel_order(self, account_id: str, order_id: str) -> None:
        raise NotImplementedError("Phase 6/7 — order routing is gated.")
