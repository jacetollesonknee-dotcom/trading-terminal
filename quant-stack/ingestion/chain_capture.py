"""Capture-forward for options chains: snapshot today, so there is a history tomorrow.

Historical option chains with open interest and greeks are what the options
backtester and the GEX analytics run on, and nobody gives them away. The
only free source is the one you build yourself by recording the live chain
every day. This module is that recorder.

It is broker-agnostic: anything satisfying :class:`~ingestion.brokers.base.BrokerClient`
works, and the broker is resolved through the registry, so the
``brokers_enabled`` feature gate applies — a disabled broker raises
:class:`~ingestion.brokers.registry.BrokerDisabledError` rather than silently
capturing nothing.

Cadence is the caller's concern (the scheduler runs this once per day after
the close; the CLI runs it on demand). The store keeps one snapshot per
underlying per expiry per day; a second capture the same day is reported as
deduplicated, not written twice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol

from config.settings import Settings
from ingestion.brokers.registry import get_broker
from ingestion.schema import OptionsChainSnapshot
from storage.store import ParquetStore, WriteResult


class ChainSource(Protocol):
    """The slice of a broker this module actually uses.

    Every :class:`~ingestion.brokers.base.BrokerClient` satisfies it
    structurally; depending on the narrower shape keeps tests honest
    (a fake needs three methods, not nine) and keeps this module from
    caring about order routing it never calls.
    """

    async def __aenter__(self) -> object: ...
    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    async def get_options_chain(
        self,
        underlying: str,
        *,
        expiration: date | None = None,
        strike_range: tuple[float, float] | None = None,
    ) -> OptionsChainSnapshot: ...


@dataclass(frozen=True)
class CaptureResult:
    """Outcome of capturing one underlying's chain."""

    underlying: str
    started_at: datetime
    write_result: WriteResult | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _normalize(underlyings: Sequence[str]) -> tuple[str, ...]:
    cleaned = tuple(u.strip().upper() for u in underlyings if u.strip())
    if not cleaned:
        msg = "underlyings must contain at least one ticker"
        raise ValueError(msg)
    return cleaned


async def capture_chain(
    broker: ChainSource, store: ParquetStore, underlying: str
) -> CaptureResult:
    """Fetch and persist one underlying's chain.

    Never raises for a bad symbol: the failure is recorded in the result so
    a batch can continue, and the caller reports it. Per-underlying
    isolation, not a silent fallback.
    """
    started = datetime.now(UTC)
    try:
        snapshot = await broker.get_options_chain(underlying)
        result = store.write_options_chain(snapshot)
    except Exception as e:
        return CaptureResult(underlying, started, None, error=f"{type(e).__name__}: {e}")
    return CaptureResult(underlying, started, result)


async def capture_chains(
    broker: ChainSource, store: ParquetStore, underlyings: Sequence[str]
) -> list[CaptureResult]:
    """Capture every underlying inside one broker session."""
    symbols = _normalize(underlyings)
    async with broker:
        return [await capture_chain(broker, store, u) for u in symbols]


def run_capture(
    settings: Settings,
    store: ParquetStore,
    underlyings: Sequence[str],
    *,
    broker_name: str = "schwab",
    broker: ChainSource | None = None,
) -> list[CaptureResult]:
    """Synchronous entry point: resolve the broker through the gate and capture.

    Args:
        settings: Process settings; supplies the ``brokers_enabled`` gate.
        store: Where snapshots are persisted.
        underlyings: Tickers to capture.
        broker_name: Registry name to resolve when ``broker`` is not given.
        broker: An already-constructed client (tests, or a caller that owns
            the lifecycle). Bypasses the registry lookup only — not the gate,
            which the caller already passed to obtain the client.

    Raises:
        BrokerDisabledError: The broker is not enabled in settings.
        BrokerUnknownError: No such broker.
        ValueError: No underlyings given.
    """
    symbols = _normalize(underlyings)
    client: ChainSource = broker if broker is not None else get_broker(broker_name, settings)
    return asyncio.run(capture_chains(client, store, symbols))
