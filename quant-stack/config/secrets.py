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
from datetime import UTC, datetime
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
    p.add_argument("--env", default="production", choices=["production", "sandbox"])
    args = p.parse_args()

    try:
        if args.set_schwab_token:
            print(f"Seeding Schwab token for env={args.env}.")
            print("Paste the OAuth JSON (single line, then enter):")
            raw = getpass.getpass(prompt="token JSON> ")
            token = SchwabToken.from_json(raw)
            set_schwab_token(token, env=args.env)
            print("OK — stored in OS keychain.")
            return 0
        if args.delete_schwab_token:
            ok = delete_schwab_token(env=args.env)
            print("Deleted." if ok else "No token to delete.")
            return 0
    except KeyringError as e:
        print(f"keyring error: {e}", file=sys.stderr)
        return 2

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
