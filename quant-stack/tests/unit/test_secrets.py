"""Tests for the keychain wrappers.

Uses keyring's in-memory backend so nothing touches the real OS store.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import keyring
import pytest
from keyring.backend import KeyringBackend

from config.secrets import (
    SchwabToken,
    delete_schwab_token,
    get_schwab_token,
    set_schwab_token,
)


class _InMemoryKeyring(KeyringBackend):
    """Trivial keyring backend for tests — never touches the real OS store."""

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self._store:
            from keyring.errors import PasswordDeleteError

            raise PasswordDeleteError
        del self._store[(service, username)]


@pytest.fixture(autouse=True)
def _isolated_keyring():  # type: ignore[no-untyped-def]
    """Swap in the in-memory backend for the duration of every test."""
    original = keyring.get_keyring()
    keyring.set_keyring(_InMemoryKeyring())
    yield
    keyring.set_keyring(original)


def _sample_token() -> SchwabToken:
    return SchwabToken(
        access_token="acc-abc",
        refresh_token="ref-xyz",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="read trade",
    )


def test_round_trip_set_get_delete() -> None:
    assert get_schwab_token(env="production") is None

    tok = _sample_token()
    set_schwab_token(tok, env="production")

    loaded = get_schwab_token(env="production")
    assert loaded is not None
    assert loaded.access_token == "acc-abc"
    assert loaded.refresh_token == "ref-xyz"

    assert delete_schwab_token(env="production") is True
    assert delete_schwab_token(env="production") is False
    assert get_schwab_token(env="production") is None


def test_envs_are_isolated() -> None:
    """production and sandbox tokens live under different service names."""
    prod = _sample_token()
    sand = SchwabToken(
        access_token="sandbox-acc",
        refresh_token="sandbox-ref",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    set_schwab_token(prod, env="production")
    set_schwab_token(sand, env="sandbox")

    assert get_schwab_token(env="production").access_token == "acc-abc"  # type: ignore[union-attr]
    assert get_schwab_token(env="sandbox").access_token == "sandbox-acc"  # type: ignore[union-attr]


def test_invalid_env_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported Schwab env"):
        get_schwab_token(env="staging")  # type: ignore[arg-type]


def test_is_expired() -> None:
    fresh = SchwabToken(
        access_token="a",
        refresh_token="r",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    stale = SchwabToken(
        access_token="a",
        refresh_token="r",
        token_type="Bearer",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    assert fresh.is_expired() is False
    assert stale.is_expired() is True
