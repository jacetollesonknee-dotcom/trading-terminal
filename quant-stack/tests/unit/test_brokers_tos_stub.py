"""TOSClient stub conforms to the BrokerClient protocol shape.

Phase 1.4a: the stub raises NotImplementedError on every method. These
tests assert the *shape* is right — methods exist, signatures match,
and the protocol's runtime_checkable check passes.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from config.settings import Settings
from ingestion.brokers.base import BrokerClient
from ingestion.brokers.tos_client import TOSClient


def test_tos_client_satisfies_protocol() -> None:
    client = TOSClient(Settings())
    assert isinstance(client, BrokerClient)
    assert client.name == "tos"


def test_tos_client_methods_are_async() -> None:
    for method in (
        "authenticate", "refresh_token",
        "get_history", "get_quote", "get_options_chain",
        "get_account", "get_positions",
        "place_order", "cancel_order",
    ):
        fn = getattr(TOSClient, method)
        assert inspect.iscoroutinefunction(fn), f"{method} must be async"


def test_tos_client_stub_methods_raise_not_implemented() -> None:
    client = TOSClient(Settings())

    async def run() -> None:
        for method, args in (
            ("authenticate", ()),
            ("refresh_token", ()),
            ("get_quote", ("NVDA",)),
            ("get_account", ("ACC-1",)),
            ("get_positions", ("ACC-1",)),
        ):
            with pytest.raises(NotImplementedError, match="Phase"):
                await getattr(client, method)(*args)

    asyncio.run(run())


def test_tos_client_history_signature_matches_protocol() -> None:
    sig_tos = inspect.signature(TOSClient.get_history)
    params = sig_tos.parameters
    assert "symbol" in params
    assert "period" in params
    assert "interval" in params
    assert params["interval"].default == "1d"
