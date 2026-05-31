"""Broker registry — name → client factory, gated by Settings.brokers_enabled.

This is the single chokepoint every caller goes through to acquire a
broker. Two reasons:

1. **Feature flag enforcement.** If ``Settings.brokers_enabled["schwab"]``
   is False, ``get_broker("schwab")`` raises ``BrokerDisabledError``
   regardless of whether a token exists. Disabled by default; you flip
   the flag in your env or via the CLI once you're ready.

2. **One place to register new brokers.** Adding a third broker later
   (Tradier, IBKR) is one entry in ``_REGISTRY`` plus a new client class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from config.settings import BrokerName, Settings
from ingestion.brokers.base import BrokerClient


class BrokerDisabledError(RuntimeError):
    """Raised when the requested broker is not enabled in Settings."""


class BrokerUnknownError(KeyError):
    """Raised when the requested broker is not registered."""


# Late-bound factories so import of this module doesn't drag in every client.
def _schwab_factory(settings: Settings) -> BrokerClient:
    from ingestion.schwab_client import SchwabClient

    return SchwabClient(settings)


def _tos_factory(settings: Settings) -> BrokerClient:
    from ingestion.brokers.tos_client import TOSClient

    return TOSClient(settings)


_REGISTRY: Final[dict[BrokerName, Callable[[Settings], BrokerClient]]] = {
    BrokerName.schwab: _schwab_factory,
    BrokerName.tos: _tos_factory,
}


def get_broker(name: str | BrokerName, settings: Settings) -> BrokerClient:
    """Resolve a broker by name. Raises if unknown or disabled.

    :param name: ``"schwab"`` or ``"tos"`` (case-insensitive string, or
        :class:`BrokerName` member).
    :param settings: process settings. Determines the gate.
    :returns: a fresh broker client instance. Caller is responsible for
        ``async with client: ...`` lifecycle.
    :raises BrokerUnknownError: name not in the registry.
    :raises BrokerDisabledError: name not in ``settings.brokers_enabled``
        or the entry is False.
    """
    # BrokerName is a StrEnum (subclass of str), so isinstance(name, BrokerName)
    # must be checked BEFORE isinstance(name, str), otherwise every enum value
    # falls through to the str branch.
    if isinstance(name, BrokerName):
        broker = name
    else:
        try:
            broker = BrokerName(name.lower())
        except ValueError as e:
            msg = f"unknown broker {name!r}. valid: {[b.value for b in BrokerName]}"
            raise BrokerUnknownError(msg) from e

    if not settings.brokers_enabled.get(broker.value, False):
        msg = (
            f"broker {broker.value!r} is not enabled. "
            f"Set brokers_enabled[{broker.value!r}]=True in your config "
            f"or via `python -m cli connect {broker.value}`."
        )
        raise BrokerDisabledError(msg)

    factory = _REGISTRY[broker]
    return factory(settings)


def list_enabled(settings: Settings) -> list[BrokerName]:
    """All brokers currently flagged True in settings."""
    return [b for b in BrokerName if settings.brokers_enabled.get(b.value, False)]


def list_registered() -> list[BrokerName]:
    """All brokers the registry knows about, enabled or not."""
    return list(_REGISTRY)
