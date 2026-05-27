"""Schwab API client — interface only.

**Phase 0 scope:** signatures + docstrings for historical pulls. Live-trading
methods (``place_order``, ``cancel_order``) are present in the surface so
callers can typecheck against them, but raise ``NotImplementedError`` with a
phase pointer until Phase 6/7 implements them.

Token storage lives in :mod:`config.secrets` (OS keychain). This module
neither reads nor writes ``.env`` files.

The client is intentionally **separate** from any Schwab integration in
Cowork. Two OAuth surfaces, two tokens, isolated blast radius — see ADR-002.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import httpx

from config.secrets import SchwabToken
from config.settings import SchwabEnv, Settings
from ingestion.schema import (
    AccountSnapshot,
    EquityBar,
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


class SchwabAPIError(RuntimeError):
    """Raised when the Schwab API returns a non-recoverable error."""


class SchwabAuthError(SchwabAPIError):
    """OAuth failed or token cannot be refreshed."""


class SchwabRateLimitError(SchwabAPIError):
    """429 from Schwab. Client backs off and retries; surfaces if persistent."""


# ─────────────────────────────────────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────────────────────────────────────


class SchwabClient:
    """Async HTTP client for the Schwab Trader API.

    Lifecycle::

        async with SchwabClient(settings) as client:
            bars = await client.get_history("NVDA", period="1y", interval="1d")

    Concurrency: one client instance per process is sufficient; the underlying
    ``httpx.AsyncClient`` is connection-pooled.

    Rate limiting: outbound requests are throttled to
    ``settings.schwab_rate_limit_rps`` (default 2 rps). On 429 the client
    sleeps with exponential backoff up to 3 retries before raising
    :class:`SchwabRateLimitError`.

    Authentication: the token is loaded from the OS keychain at
    :meth:`authenticate` time. If expired, :meth:`refresh_token` is invoked
    automatically. The refresh path uses Schwab's refresh-token grant; if
    that also fails, :class:`SchwabAuthError` is raised and the caller must
    re-run the OAuth flow out-of-band.
    """

    def __init__(self, settings: Settings) -> None:
        """Construct a client bound to a Settings instance.

        :param settings: process settings. Determines API base URL (prod vs
            sandbox), request timeout, and rate-limit ceiling.
        """
        self._settings = settings
        self._base_url = _BASE_URLS[settings.schwab_env]
        self._token: SchwabToken | None = None
        self._http: httpx.AsyncClient | None = None
        raise NotImplementedError("Phase 0 — interface only.")

    # ── async context manager ──────────────────────────────────────────

    async def __aenter__(self) -> SchwabClient:
        """Open the underlying HTTP client and load the OAuth token."""
        raise NotImplementedError("Phase 0 — interface only.")

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        """Close the underlying HTTP client cleanly."""
        raise NotImplementedError("Phase 0 — interface only.")

    # ── auth ───────────────────────────────────────────────────────────

    async def authenticate(self) -> None:
        """Load token from keychain. Refresh if within 60s of expiry.

        :raises SchwabAuthError: no token in keychain, or refresh failed.
            The caller must run the OAuth flow out-of-band and seed the
            token via ``python -m config.secrets --set-schwab-token``.
        """
        raise NotImplementedError("Phase 0 — interface only.")

    async def refresh_token(self) -> SchwabToken:
        """Exchange the refresh_token for a fresh access_token.

        Persists the new token back to the OS keychain on success.

        :returns: the new :class:`SchwabToken`.
        :raises SchwabAuthError: Schwab rejected the refresh.
        """
        raise NotImplementedError("Phase 0 — interface only.")

    # ── historical pulls (Phase 0 priority) ────────────────────────────

    async def get_history(
        self,
        symbol: str,
        *,
        period: Literal["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd"],
        interval: Literal["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"] = "1d",
    ) -> list[EquityBar]:
        """Pull historical OHLCV bars for a single equity.

        Bars are returned in chronological order, oldest first, with every
        record tagged with the bar's session-close time as ``as_of``
        (UTC). Schwab's intraday data goes back only ~6 months; daily goes
        back further. Caller should fall back to Yahoo for deep history if
        the response is short.

        :param symbol: equity ticker, uppercase.
        :param period: lookback window. ``ytd`` is year-to-date.
        :param interval: bar resolution. ``1m`` and ``5m`` are intraday only.
        :returns: list of :class:`EquityBar` in chronological order.
        :raises SchwabAPIError: unrecoverable API failure.
        :raises SchwabRateLimitError: 429 after 3 backoff retries.
        """
        raise NotImplementedError("Phase 0 — interface only.")

    async def get_quote(self, symbol: str) -> EquityBar:
        """Snapshot quote as a single 1d bar with ``as_of`` = now (UTC).

        Used by the Phase 0 smoke test to confirm OAuth + connectivity.
        """
        raise NotImplementedError("Phase 0 — interface only.")

    async def get_options_chain(
        self,
        underlying: str,
        *,
        expiration: date | None = None,
        strike_range: tuple[float, float] | None = None,
    ) -> OptionsChainSnapshot:
        """Pull a current options chain snapshot for ``underlying``.

        Schwab returns chains with Greeks, IV, OI, and volume populated by
        their own pricing service. We persist the full chain (no filtering
        beyond ``expiration`` and ``strike_range``) so backtests can replay
        whatever filter logic future signals dream up.

        :param underlying: equity ticker, uppercase.
        :param expiration: if given, only contracts expiring on this date.
            If None, all expirations within Schwab's default window.
        :param strike_range: ``(low, high)`` strike filter, inclusive.
        :returns: one :class:`OptionsChainSnapshot` with all matching contracts.
        """
        raise NotImplementedError("Phase 0 — interface only.")

    # ── account / positions (Phase 5+) ─────────────────────────────────

    async def get_account(self, account_id: str) -> AccountSnapshot:
        """Fetch one account's current state. Phase 5 onward."""
        raise NotImplementedError("Phase 5 — risk + sizing.")

    async def get_positions(self, account_id: str) -> list[Position]:
        """Fetch open positions for one account. Phase 5 onward."""
        raise NotImplementedError("Phase 5 — risk + sizing.")

    # ── orders (Phase 6 paper, Phase 7 live) ───────────────────────────

    async def place_order(self, order: Order) -> Order:
        """Submit an order. Paper or live depending on settings.

        Hard guardrails enforced upstream in ``execution/order_generator.py``
        (naked-only) and ``backend/routers/orders.py`` (live-orders-armed
        check). This method does not relax them — it raises immediately if
        ``settings.live_orders_armed`` is False and ``source`` is ``"live"``.

        Phase 6 implements the paper path; Phase 7 implements live.
        """
        raise NotImplementedError("Phase 6 (paper) / Phase 7 (live).")

    async def cancel_order(self, account_id: str, order_id: str) -> None:
        """Cancel a working order. Phase 6 onward."""
        raise NotImplementedError("Phase 6 (paper) / Phase 7 (live).")
