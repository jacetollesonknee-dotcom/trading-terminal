"""`python -m cli connect ...` smoke tests.

Uses the in-memory keyring backend (lifted from test_secrets.py pattern)
so nothing touches the real OS keychain.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import httpx
import keyring
import pytest
import respx
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

from cli import connect
from cli.__main__ import main
from config.secrets import (
    SchwabAppCredentials,
    SchwabToken,
    get_schwab_token,
    set_schwab_app_credentials,
    set_schwab_token,
)
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
#  connect (paste-the-redirect-URL OAuth flow)
# ─────────────────────────────────────────────────────────────────────────

_REDIRECT = "https://127.0.0.1:8182"


def _seed_app_creds(env: str = "production") -> None:
    set_schwab_app_credentials(SchwabAppCredentials("KEY", "SECRET"), env=env)


def test_connect_without_app_credentials_explains_what_to_do(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = connect.cmd_connect(BrokerName.schwab, env="production")
    assert code == 2  # advisory
    out = capsys.readouterr().out
    assert "No Schwab app credentials" in out
    assert "--set-schwab-app" in out


@respx.mock(base_url="https://api.schwabapi.com")
def test_connect_full_flow_stores_token(
    respx_mock: respx.MockRouter, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_app_creds()
    exchange = respx_mock.post("/v1/oauth/token").mock(
        return_value=httpx.Response(200, json={
            "access_token": "A", "refresh_token": "R", "token_type": "Bearer",
            "expires_in": 1800, "scope": "api",
        })
    )
    pasted = f"{_REDIRECT}/?code=C0DE%40abc&session=xyz"
    code = connect.cmd_connect(BrokerName.schwab, env="production", prompt=lambda _p: pasted)
    assert code == 0
    assert exchange.called
    assert b"code=C0DE%40abc" in exchange.calls[0].request.content  # url-decoded then re-encoded
    stored = get_schwab_token(env="production")
    assert stored is not None
    assert stored.access_token == "A"
    out = capsys.readouterr().out
    assert "/v1/oauth/authorize?" in out  # the URL to open was printed
    assert "BROKERS_ENABLED" in out  # and the exact line to enable the flag


def test_connect_rejects_redirect_without_code(capsys: pytest.CaptureFixture[str]) -> None:
    _seed_app_creds()
    code = connect.cmd_connect(
        BrokerName.schwab, env="production", prompt=lambda _p: f"{_REDIRECT}/?error=denied"
    )
    assert code == 2
    assert "no ?code=" in capsys.readouterr().out
    assert get_schwab_token(env="production") is None


@respx.mock(base_url="https://api.schwabapi.com")
def test_connect_reports_schwab_rejection(
    respx_mock: respx.MockRouter, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_app_creds()
    respx_mock.post("/v1/oauth/token").mock(return_value=httpx.Response(400, text="bad"))
    code = connect.cmd_connect(
        BrokerName.schwab, env="production", prompt=lambda _p: f"{_REDIRECT}/?code=X"
    )
    assert code == 2
    assert "rejected" in capsys.readouterr().out
    assert get_schwab_token(env="production") is None


def test_connect_does_not_flip_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Connect must not silently enable a broker; the flag lives in .env."""
    settings = Settings()
    assert settings.brokers_enabled["schwab"] is False
    connect.cmd_connect(BrokerName.schwab, env="production")
    # Re-read settings — flag still False.
    assert Settings().brokers_enabled["schwab"] is False


def test_code_from_redirect() -> None:
    assert connect._code_from_redirect(f"{_REDIRECT}/?code=abc%2Fdef&x=1") == "abc/def"
    assert connect._code_from_redirect(f"{_REDIRECT}/") is None
    assert connect._code_from_redirect("not a url") is None


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
