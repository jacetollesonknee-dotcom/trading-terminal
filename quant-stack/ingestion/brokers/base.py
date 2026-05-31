"""Broker interface — structural protocol that every broker client conforms to.

The Protocol is intentionally minimal in v1: enough surface for callers to
ask "give me a broker by name" without caring whether it's Schwab, TOS, or
something added later. As more broker types land, this surface grows.

Schwab and TOS technically sit on the same Schwab Trader API since the
Ameritrade consolidation; they differ only in account scope. So they share
~95% of implementation but each gets its own client class for clarity (and
for the feature-flag toggle).
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Protocol, runtime_checkable

from ingestion.schema import (
    AccountSnapshot,
    EquityBar,
    OptionsChainSnapshot,
    Order,
    Position,
)


@runtime_checkable
class BrokerClient(Protocol):
    """The methods every broker client must expose.

    Async by convention. Concrete clients use httpx.AsyncClient under the
    hood. Errors raised:

    - ``BrokerAuthError`` — token absent, refresh failed, OAuth needed.
    - ``BrokerAPIError`` — non-recoverable upstream failure.
    - ``BrokerRateLimitError`` — 429 after retries exhausted.

    None of these are imported here to avoid pulling each client's error
    module into every caller. The Protocol describes shape, not exceptions.
    """

    name: str  # "schwab" | "tos"

    async def __aenter__(self) -> BrokerClient: ...
    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    async def authenticate(self) -> None:
        """Load token from keychain; refresh if within skew of expiry."""
        ...

    async def refresh_token(self) -> None:
        """Exchange the refresh token for a new access token."""
        ...

    async def get_history(
        self,
        symbol: str,
        *,
        period: Literal["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd"],
        interval: Literal["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"] = "1d",
    ) -> list[EquityBar]:
        """Historical OHLCV bars."""
        ...

    async def get_quote(self, symbol: str) -> EquityBar:
        """Single latest bar with as_of = now (UTC)."""
        ...

    async def get_options_chain(
        self,
        underlying: str,
        *,
        expiration: date | None = None,
        strike_range: tuple[float, float] | None = None,
    ) -> OptionsChainSnapshot:
        """Current options chain snapshot."""
        ...

    async def get_account(self, account_id: str) -> AccountSnapshot:
        """Account state — cash, equity, margin, buying power."""
        ...

    async def get_positions(self, account_id: str) -> list[Position]:
        """Open positions for the account."""
        ...

    async def place_order(self, order: Order) -> Order:
        """Submit an order. Live orders only when Settings.live_orders_armed."""
        ...

    async def cancel_order(self, account_id: str, order_id: str) -> None:
        """Cancel a working order."""
        ...
