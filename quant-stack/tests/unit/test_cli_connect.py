"""`python -m cli connect ...` smoke tests.

Uses the in-memory keyring backend (lifted from test_secrets.py pattern)
so nothing touches the real OS keychain.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

from cli import connect
from cli.__main__ import main
from config.secrets import SchwabToken, set_schwab_token
from config.settings import BrokerName, Settings


class _InMemoryKeyring(KeyringBackend):
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
    original = keyring.get_keyring()
    keyring.set_keyring(_InMemoryKeyring())
    yield
    keyring.set_keyring(original)


def _seed_token(env: str = "production") -> None:
    set_schwab_token(
        SchwabToken(
            access_token="acc",
            refresh_token="ref",
            token_type="Bearer",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        ),
        env=env,
    )


# ─────────────────────────────────────────────────────────────────────────
#  status
# ─────────────────────────────────────────────────────────────────────────

def test_status_prints_one_line_per_broker(capsys: pytest.CaptureFixture[str]) -> None:
    code = connect.cmd_status(Settings())
    assert code == 0
    out = capsys.readouterr().out
    for broker in BrokerName:
        assert broker.value in out
    assert "enabled" in out


def test_status_reflects_token_presence(capsys: pytest.CaptureFixture[str]) -> None:
    _seed_token()
    connect.cmd_status(Settings())
    out = capsys.readouterr().out
    # token-in-keychain column is "True" for schwab/tos once token is seeded
    # (both share the Schwab keychain entry in this stub phase).
    lines = [line for line in out.splitlines() if line.startswith(("schwab", "tos"))]
    assert all("True" in line for line in lines)


# ─────────────────────────────────────────────────────────────────────────
#  connect (stub)
# ─────────────────────────────────────────────────────────────────────────

def test_connect_returns_advisory_exit_when_oauth_not_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = connect.cmd_connect(BrokerName.schwab, env="production")
    assert code == 2  # advisory
    out = capsys.readouterr().out
    assert "Phase 1.4a" in out
    assert "set-schwab-token" in out


def test_connect_does_not_flip_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """The stub must not silently enable a broker we cannot actually talk to."""
    settings = Settings()
    assert settings.brokers_enabled["schwab"] is False
    connect.cmd_connect(BrokerName.schwab, env="production")
    # Re-read settings — flag still False.
    assert Settings().brokers_enabled["schwab"] is False


# ─────────────────────────────────────────────────────────────────────────
#  disconnect
# ─────────────────────────────────────────────────────────────────────────

def test_disconnect_removes_token(capsys: pytest.CaptureFixture[str]) -> None:
    _seed_token()
    code = connect.cmd_disconnect(BrokerName.schwab, env="production")
    assert code == 0
    out = capsys.readouterr().out
    assert "Deleted" in out


def test_disconnect_when_no_token_present(capsys: pytest.CaptureFixture[str]) -> None:
    code = connect.cmd_disconnect(BrokerName.schwab, env="production")
    assert code == 0
    out = capsys.readouterr().out
    assert "No" in out and "to delete" in out


# ─────────────────────────────────────────────────────────────────────────
#  Top-level dispatch via `python -m cli`
# ─────────────────────────────────────────────────────────────────────────

def test_main_status_default(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["connect"])
    assert code == 0
    out = capsys.readouterr().out
    assert "broker" in out and "enabled" in out


def test_main_connect_unknown_broker_argparse_rejects(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["connect", "interactive_brokers"])
