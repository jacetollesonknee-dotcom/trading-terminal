"""OS-keychain wrappers for Schwab OAuth tokens.

Secrets never live in ``.env`` and never enter the git tree. On Windows the
backing store is Windows Credential Manager; on macOS, Keychain; on Linux,
Secret Service (e.g. GNOME Keyring).

The engine does not call Anthropic or any other LLM provider; that lives in
Cowork. There is intentionally no ``get_anthropic_key`` here (ADR-002).

Usage::

    from config.secrets import get_schwab_token, set_schwab_token

    token = get_schwab_token(env="production")
    if token is None:
        raise RuntimeError("Schwab OAuth not seeded; run config.secrets CLI.")

Or from the shell (one-time seeding)::

    python -m config.secrets --set-schwab-token --env production
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

# Service names — namespace by app + env so production and sandbox tokens
# never collide.
_SERVICE_PREFIX: Final[str] = "quant-stack"


def _schwab_service(env: str) -> str:
    """Keychain service identifier for the Schwab OAuth token in ``env``."""
    if env not in ("production", "sandbox"):
        msg = f"unsupported Schwab env: {env!r}"
        raise ValueError(msg)
    return f"{_SERVICE_PREFIX}.schwab.{env}"


_SCHWAB_USER: Final[str] = "oauth_token"
_SCHWAB_APP_USER: Final[str] = "app_credentials"


@dataclass(frozen=True, slots=True)
class SchwabAppCredentials:
    """The app key + secret from developer.schwab.com.

    Needed to refresh tokens (Basic auth on the token endpoint) and to run
    the authorization flow. Stored in the keychain like the token; never in
    ``.env``.
    """

    app_key: str
    app_secret: str

    def to_json(self) -> str:
        return json.dumps({"app_key": self.app_key, "app_secret": self.app_secret})

    @classmethod
    def from_json(cls, raw: str) -> SchwabAppCredentials:
        data = json.loads(raw)
        return cls(app_key=str(data["app_key"]), app_secret=str(data["app_secret"]))


@dataclass(frozen=True, slots=True)
class SchwabToken:
    """Schwab OAuth credential bundle.

    Persisted in the OS keychain as a JSON blob under
    ``quant-stack.schwab.<env>``.
    """

    access_token: str
    refresh_token: str
    token_type: str
    expires_at: datetime  # UTC
    scope: str | None = None

    def to_json(self) -> str:
        """Serialize to the on-disk keychain format."""
        return json.dumps(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "token_type": self.token_type,
                "expires_at": self.expires_at.isoformat(),
                "scope": self.scope,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> SchwabToken:
        """Deserialize from the keychain. Strict — fails loudly on bad shape."""
        data = json.loads(raw)
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            token_type=str(data["token_type"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            scope=(str(data["scope"]) if data.get("scope") else None),
        )

    def is_expired(self, *, skew_seconds: int = 60) -> bool:
        """True if the token expires within ``skew_seconds`` from now (UTC)."""
        now = datetime.now(UTC)
        return (self.expires_at - now).total_seconds() <= skew_seconds

    @classmethod
    def from_oauth_response(
        cls, data: dict[str, object], *, issued_at: datetime | None = None
    ) -> SchwabToken:
        """Build from what Schwab's token endpoint returns, or from a token file.

        Accepts three shapes, strictly:

        - The raw endpoint response: ``access_token, refresh_token, token_type,
          expires_in`` (seconds from ``issued_at``, default now).
        - The same with ``expires_at`` as a Unix epoch (what ``schwab-py``
          adds).
        - ``schwab-py``'s token file, which wraps the above under ``"token"``.
        """
        inner = data.get("token")
        if isinstance(inner, dict):
            return cls.from_oauth_response(inner, issued_at=issued_at)
        if "expires_at" in data:
            raw = data["expires_at"]
            if isinstance(raw, int | float):
                expires_at = datetime.fromtimestamp(raw, tz=UTC)
            else:
                expires_at = datetime.fromisoformat(str(raw))
        elif "expires_in" in data:
            start = issued_at or datetime.now(UTC)
            expires_at = start + timedelta(seconds=float(str(data["expires_in"])))
        else:
            msg = "token response has neither expires_in nor expires_at"
            raise ValueError(msg)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        scope = data.get("scope")
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            token_type=str(data.get("token_type", "Bearer")),
            expires_at=expires_at,
            scope=(str(scope) if scope else None),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — Schwab
# ─────────────────────────────────────────────────────────────────────────────


def get_schwab_token(*, env: str = "production") -> SchwabToken | None:
    """Read the Schwab OAuth token for ``env`` from the OS keychain.

    Returns ``None`` if no token has been seeded. Does NOT auto-refresh — the
    Schwab client owns refresh logic so the flow stays explicit.
    """
    raw = keyring.get_password(_schwab_service(env), _SCHWAB_USER)
    if raw is None:
        return None
    return SchwabToken.from_json(raw)


def set_schwab_token(token: SchwabToken, *, env: str = "production") -> None:
    """Write (or replace) the Schwab OAuth token in the OS keychain."""
    keyring.set_password(_schwab_service(env), _SCHWAB_USER, token.to_json())


def delete_schwab_token(*, env: str = "production") -> bool:
    """Remove the Schwab token from the keychain. Returns False if absent."""
    try:
        keyring.delete_password(_schwab_service(env), _SCHWAB_USER)
        return True
    except PasswordDeleteError:
        return False


def get_schwab_app_credentials(*, env: str = "production") -> SchwabAppCredentials | None:
    """Read the app key + secret for ``env``; ``None`` if never seeded."""
    raw = keyring.get_password(_schwab_service(env), _SCHWAB_APP_USER)
    if raw is None:
        return None
    return SchwabAppCredentials.from_json(raw)


def set_schwab_app_credentials(creds: SchwabAppCredentials, *, env: str = "production") -> None:
    """Write (or replace) the app key + secret in the OS keychain."""
    keyring.set_password(_schwab_service(env), _SCHWAB_APP_USER, creds.to_json())


def delete_schwab_app_credentials(*, env: str = "production") -> bool:
    """Remove the app credentials. Returns False if absent."""
    try:
        keyring.delete_password(_schwab_service(env), _SCHWAB_APP_USER)
        return True
    except PasswordDeleteError:
        return False


def app_credentials_from_env_file(path: Path) -> SchwabAppCredentials:
    """Read the app key + secret from a ``KEY=VALUE`` file (e.g. the terminal's ``.env``).

    Accepts ``SCHWAB_API_KEY``/``SCHWAB_SECRET`` (the terminal app's names) or
    ``SCHWAB_APP_KEY``/``SCHWAB_APP_SECRET``. This is a one-time migration
    INTO the keychain; the ``.env`` is read, never written.

    Raises:
        ValueError: The file lacks a key or a secret.
    """
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip().upper()] = value.strip().strip("\"'")
    app_key = values.get("SCHWAB_API_KEY") or values.get("SCHWAB_APP_KEY") or ""
    secret = values.get("SCHWAB_SECRET") or values.get("SCHWAB_APP_SECRET") or ""
    if not app_key or not secret:
        msg = f"{path}: need SCHWAB_API_KEY and SCHWAB_SECRET (or SCHWAB_APP_KEY/SCHWAB_APP_SECRET)"
        raise ValueError(msg)
    return SchwabAppCredentials(app_key=app_key, app_secret=secret)


def import_token_file(path: Path, *, env: str = "production") -> SchwabToken:
    """Seed the keychain from a token file already on disk (e.g. ``schwab-py``'s).

    Reads the JSON, accepts any shape :meth:`SchwabToken.from_oauth_response`
    does, stores it, and returns it. The file is not modified or deleted.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path}: expected a JSON object"
        raise ValueError(msg)
    token = SchwabToken.from_oauth_response(data)
    set_schwab_token(token, env=env)
    return token


# ─────────────────────────────────────────────────────────────────────────────
#  CLI — one-time seeding from the shell
# ─────────────────────────────────────────────────────────────────────────────


def _cli() -> int:
    p = argparse.ArgumentParser(
        prog="config.secrets",
        description="Seed the OS keychain with Schwab OAuth tokens.",
    )
    p.add_argument("--set-schwab-token", action="store_true")
    p.add_argument("--delete-schwab-token", action="store_true")
    p.add_argument(
        "--import-token-file",
        metavar="PATH",
        help=(
            "Seed the token from a JSON file on disk "
            "(raw OAuth response or schwab-py token file)."
        ),
    )
    p.add_argument(
        "--set-schwab-app",
        action="store_true",
        help="Store the app key + secret from developer.schwab.com (prompts; nothing echoed).",
    )
    p.add_argument(
        "--set-schwab-app-from-env-file",
        metavar="PATH",
        help="Seed the app key + secret from an existing KEY=VALUE file (e.g. terminal/.env).",
    )
    p.add_argument("--delete-schwab-app", action="store_true")
    p.add_argument("--env", default="production", choices=["production", "sandbox"])
    args = p.parse_args()

    # One handler per flag; the first flag set wins.
    handlers = (
        (args.set_schwab_token, _cli_set_token),
        (bool(args.import_token_file), _cli_import_token),
        (args.set_schwab_app, _cli_set_app),
        (bool(args.set_schwab_app_from_env_file), _cli_set_app_from_env_file),
        (args.delete_schwab_app, _cli_delete_app),
        (args.delete_schwab_token, _cli_delete_token),
    )
    try:
        for chosen, handler in handlers:
            if chosen:
                return handler(args)
    except KeyringError as e:
        print(f"keyring error: {e}", file=sys.stderr)
        return 2

    p.print_help()
    return 1


def _cli_set_token(args: argparse.Namespace) -> int:
    print(f"Seeding Schwab token for env={args.env}.")
    print("Paste the OAuth JSON (single line, then enter):")
    raw = getpass.getpass(prompt="token JSON> ")
    set_schwab_token(SchwabToken.from_json(raw), env=args.env)
    print("OK — stored in OS keychain.")
    return 0


def _cli_import_token(args: argparse.Namespace) -> int:
    token = import_token_file(Path(args.import_token_file), env=args.env)
    state = "EXPIRED — the client will refresh it" if token.is_expired() else "valid"
    print(f"OK — imported token for env={args.env} (expires {token.expires_at}, {state}).")
    return 0


def _cli_set_app(args: argparse.Namespace) -> int:
    print(f"Seeding Schwab app credentials for env={args.env}.")
    key = getpass.getpass(prompt="app key> ").strip()
    secret = getpass.getpass(prompt="app secret> ").strip()
    if not key or not secret:
        print("Both values are required.", file=sys.stderr)
        return 1
    set_schwab_app_credentials(SchwabAppCredentials(key, secret), env=args.env)
    print("OK — stored in OS keychain.")
    return 0


def _cli_set_app_from_env_file(args: argparse.Namespace) -> int:
    creds = app_credentials_from_env_file(Path(args.set_schwab_app_from_env_file))
    set_schwab_app_credentials(creds, env=args.env)
    print(f"OK — app credentials for env={args.env} stored in OS keychain "
          f"(key ends ...{creds.app_key[-4:]}).")
    return 0


def _cli_delete_app(args: argparse.Namespace) -> int:
    ok = delete_schwab_app_credentials(env=args.env)
    print("Deleted." if ok else "No app credentials to delete.")
    return 0


def _cli_delete_token(args: argparse.Namespace) -> int:
    ok = delete_schwab_token(env=args.env)
    print("Deleted." if ok else "No token to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
