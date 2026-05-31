"""ThinkorSwim client — stub conforming to :class:`BrokerClient`.

TOS web/mobile and Schwab both sit on the Schwab Trader API since the
Ameritrade consolidation. The only thing that differs is account scope
and a couple of TOS-specific account ID formats. So this client will
ultimately be ~95% the same as ``SchwabClient`` but is kept as its own
class for clarity and for the per-broker feature gate.

**Phase 1.4a scope:** stub. Every method raises ``NotImplementedError``
with a phase pointer. The class exists so the registry, the CLI, and
the type-checker have something concrete to point at. Real bodies land
when the operator has Schwab OAuth credentials.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from config.settings import SchwabEnv, Settings
from ingestion.schema import (
    AccountSnapshot,
    EquityBar,
    OptionsChainSnapshot,
    Order,
    Position,
)

_TOS_BASE_URLS: dict[SchwabEnv, str] = {
    # TOS uses the same Schwab API base URLs; the account-id format and
    # scope are what distinguish a TOS-linked account from a Schwab-linked one.
    SchwabEnv.production: "https://api.schwabapi.com",
    SchwabEnv.sandbox: "https://api-sandbox.schwabapi.com",
}


class TOSClient:
    """ThinkorSwim broker client. Stub until OAuth credentials are seeded."""

    name = "tos"

    def __init__(self, settings: Settings) -> None:
        """Bind to settings; nothing networked yet."""
        self._settings = settings
        self._base_url = _TOS_BASE_URLS[settings.schwab_env]

    async def __aenter__(self) -> TOSClient:
        raise NotImplementedError("Phase 1.4a — stub. OAuth lands when keys are ready.")

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        raise NotImplementedError("Phase 1.4a — stub.")

    async def authenticate(self) -> None:
        raise NotImplementedError("Phase 1.4a — stub.")

    async def refresh_token(self) -> None:
        raise NotImplementedError("Phase 1.4a — stub.")

    async def get_history(
        self,
        symbol: str,
        *,
        period: Literal["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd"],
        interval: Literal["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"] = "1d",
    ) -> list[EquityBar]:
        raise NotImplementedError("Phase 1.4a — stub.")

    async def get_quote(self, symbol: str) -> EquityBar:
        raise NotImplementedError("Phase 1.4a — stub.")

    async def get_options_chain(
        self,
        underlying: str,
        *,
        expiration: date | None = None,
        strike_range: tuple[float, float] | None = None,
    ) -> OptionsChainSnapshot:
        raise NotImplementedError("Phase 1.4a — stub.")

    async def get_account(self, account_id: str) -> AccountSnapshot:
        raise NotImplementedError("Phase 5 — risk + sizing.")

    async def get_positions(self, account_id: str) -> list[Position]:
        raise NotImplementedError("Phase 5 — risk + sizing.")

    async def place_order(self, order: Order) -> Order:
        raise NotImplementedError("Phase 6 (paper) / Phase 7 (live).")

    async def cancel_order(self, account_id: str, order_id: str) -> None:
        raise NotImplementedError("Phase 6 (paper) / Phase 7 (live).")
