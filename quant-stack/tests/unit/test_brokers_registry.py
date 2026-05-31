"""Broker registry tests: gating, lookup, error paths."""

from __future__ import annotations

import pytest

from config.settings import BrokerName, Settings
from ingestion.brokers.registry import (
    BrokerDisabledError,
    BrokerUnknownError,
    get_broker,
    list_enabled,
    list_registered,
)


def _settings(*, enabled: dict[str, bool] | None = None) -> Settings:
    """Build a Settings instance with a specific brokers_enabled mapping."""
    if enabled is None:
        enabled = {b.value: False for b in BrokerName}
    return Settings(brokers_enabled=enabled)


def test_disabled_broker_raises() -> None:
    s = _settings(enabled={"schwab": False, "tos": False})
    with pytest.raises(BrokerDisabledError, match="not enabled"):
        get_broker("schwab", s)
    with pytest.raises(BrokerDisabledError, match="not enabled"):
        get_broker("tos", s)


def test_enabled_broker_returns_client_instance() -> None:
    s = _settings(enabled={"schwab": True, "tos": False})
    client = get_broker("schwab", s)
    assert client.name == "schwab"


def test_enabled_tos_returns_tos_client() -> None:
    s = _settings(enabled={"schwab": False, "tos": True})
    client = get_broker("tos", s)
    assert client.name == "tos"


def test_unknown_broker_raises() -> None:
    s = _settings(enabled={"interactive_brokers": True})
    with pytest.raises(BrokerUnknownError, match="unknown broker"):
        get_broker("interactive_brokers", s)


def test_accepts_brokername_enum_directly() -> None:
    s = _settings(enabled={"schwab": True})
    client = get_broker(BrokerName.schwab, s)
    assert client.name == "schwab"


def test_case_insensitive_string_name() -> None:
    s = _settings(enabled={"schwab": True})
    assert get_broker("SCHWAB", s).name == "schwab"
    assert get_broker("Schwab", s).name == "schwab"


def test_list_enabled_returns_only_enabled() -> None:
    s = _settings(enabled={"schwab": True, "tos": False})
    enabled = list_enabled(s)
    assert BrokerName.schwab in enabled
    assert BrokerName.tos not in enabled


def test_list_registered_includes_all_known_brokers() -> None:
    assert set(list_registered()) == set(BrokerName)


def test_disable_default_is_off_for_every_broker() -> None:
    """Defaults must NOT silently enable any broker."""
    s = Settings()  # defaults
    for b in BrokerName:
        assert s.brokers_enabled.get(b.value) is False, (
            f"broker {b.value} must default to disabled"
        )
