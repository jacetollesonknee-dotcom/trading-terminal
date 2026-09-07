"""Tests for the keychain wrappers.

Uses keyring's in-memory backend so nothing touches the real OS store.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

from config.secrets import (
    SchwabAppCredentials,
    SchwabToken,
    app_credentials_from_env_file,
    delete_schwab_app_credentials,
    delete_schwab_token,
    get_schwab_app_credentials,
    get_schwab_token,
    import_token_file,
    set_schwab_app_credentials,
    set_schwab_token,
)


class _InMemoryKeyring(KeyringBackend):
    """Trivial keyring backend for tests — never touches the real OS store."""

    priority = 1

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self._store:
            raise PasswordDeleteError
        del self._store[(service, username)]


@pytest.fixture(autouse=True)
def _isolated_keyring() -> Iterator[None]:
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
        get_schwab_token(env="staging")


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


# ─────────────────────────────────────────────────────────────────────────────
#  App credentials
# ─────────────────────────────────────────────────────────────────────────────


def test_app_credentials_round_trip() -> None:
    assert get_schwab_app_credentials(env="production") is None
    set_schwab_app_credentials(SchwabAppCredentials("KEY", "SECRET"), env="production")
    loaded = get_schwab_app_credentials(env="production")
    assert loaded is not None
    assert loaded.app_key == "KEY"
    assert loaded.app_secret == "SECRET"
    # Separate keychain entry from the token.
    assert get_schwab_token(env="production") is None
    assert delete_schwab_app_credentials(env="production") is True
    assert delete_schwab_app_credentials(env="production") is False


# ─────────────────────────────────────────────────────────────────────────────
#  Token response shapes
# ─────────────────────────────────────────────────────────────────────────────


def test_from_oauth_response_with_expires_in() -> None:
    issued = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)
    tok = SchwabToken.from_oauth_response(
        {"access_token": "A", "refresh_token": "R", "token_type": "Bearer", "expires_in": 1800},
        issued_at=issued,
    )
    assert tok.expires_at == issued + timedelta(seconds=1800)
    assert tok.scope is None


def test_from_oauth_response_with_epoch_expires_at() -> None:
    tok = SchwabToken.from_oauth_response(
        {"access_token": "A", "refresh_token": "R", "expires_at": 1_800_000_000, "scope": "api"}
    )
    assert tok.expires_at == datetime.fromtimestamp(1_800_000_000, tz=UTC)
    assert tok.token_type == "Bearer"  # defaulted
    assert tok.scope == "api"


def test_from_oauth_response_schwab_py_wrapper() -> None:
    wrapped = {
        "creation_timestamp": 1_700_000_000,
        "token": {
            "access_token": "A", "refresh_token": "R", "token_type": "Bearer",
            "expires_in": 1800, "expires_at": 1_800_000_000,
        },
    }
    tok = SchwabToken.from_oauth_response(wrapped)
    assert tok.access_token == "A"
    assert tok.expires_at == datetime.fromtimestamp(1_800_000_000, tz=UTC)


def test_from_oauth_response_requires_an_expiry() -> None:
    with pytest.raises(ValueError, match="expires"):
        SchwabToken.from_oauth_response({"access_token": "A", "refresh_token": "R"})


def test_import_token_file(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    path.write_text(
        '{"token": {"access_token": "FILE", "refresh_token": "R", '
        '"token_type": "Bearer", "expires_at": 1800000000}}',
        encoding="utf-8",
    )
    tok = import_token_file(path, env="sandbox")
    assert tok.access_token == "FILE"
    stored = get_schwab_token(env="sandbox")
    assert stored is not None and stored.access_token == "FILE"
    assert get_schwab_token(env="production") is None
    assert path.exists()  # the file is left alone


def test_import_token_file_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        import_token_file(path)


# ─────────────────────────────────────────────────────────────────────────────
#  App credentials from an existing .env
# ─────────────────────────────────────────────────────────────────────────────


def test_app_credentials_from_terminal_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# terminal app keys\n"
        "SCHWAB_API_KEY=abc123\n"
        'SCHWAB_SECRET="s3cret"\n'
        "ANTHROPIC_API_KEY=sk-ant-x\n"
        "PAPER_TRADING=true\n",
        encoding="utf-8",
    )
    creds = app_credentials_from_env_file(env)
    assert creds.app_key == "abc123"
    assert creds.app_secret == "s3cret"  # quotes stripped
    assert env.read_text(encoding="utf-8").startswith("# terminal")  # never written


def test_app_credentials_from_env_file_accepts_alternate_names(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("SCHWAB_APP_KEY=k\nSCHWAB_APP_SECRET=s\n", encoding="utf-8")
    creds = app_credentials_from_env_file(env)
    assert (creds.app_key, creds.app_secret) == ("k", "s")


def test_app_credentials_from_env_file_requires_both(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("SCHWAB_API_KEY=k\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SCHWAB_SECRET"):
        app_credentials_from_env_file(env)
