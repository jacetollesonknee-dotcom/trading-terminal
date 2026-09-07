"""`python -m cli connect ...` — manage broker connections.

Three subcommands:

    connect status                 List brokers and their connection state.
    connect <broker> [--env ENV]   Begin OAuth flow + flip the feature flag.
    connect <broker> --disconnect  Delete token + flip flag back off.

The connect flow is the paste-the-redirect-URL variant of OAuth: no local
web server, no browser automation. Prerequisite: app credentials in the
keychain (``python -m config.secrets --set-schwab-app``). If you already
have a token file from another tool (e.g. schwab-py), skip the flow and
``python -m config.secrets --import-token-file <path>`` instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Final

from config.secrets import delete_schwab_token, get_schwab_token
from config.settings import BrokerName, Settings, get_settings
from ingestion.brokers.registry import list_registered

_VALID_ENVS: Final[tuple[str, ...]] = ("production", "sandbox")


# ─────────────────────────────────────────────────────────────────────────────
#  argparse wiring
# ─────────────────────────────────────────────────────────────────────────────


def add_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "connect",
        help="Manage broker connections (Schwab, TOS).",
    )
    p.add_argument(
        "broker",
        nargs="?",
        choices=["status", *(b.value for b in BrokerName)],
        default="status",
        help='Broker name, or "status" to list all (default).',
    )
    p.add_argument("--env", default="production", choices=_VALID_ENVS)
    p.add_argument(
        "--disconnect",
        action="store_true",
        help="Remove the token from the OS keychain and disable the flag.",
    )


def dispatch(args: argparse.Namespace) -> int:
    if args.broker == "status":
        return cmd_status(get_settings())
    if args.disconnect:
        return cmd_disconnect(BrokerName(args.broker), args.env)
    return cmd_connect(BrokerName(args.broker), args.env)


# ─────────────────────────────────────────────────────────────────────────────
#  Commands
# ─────────────────────────────────────────────────────────────────────────────


def cmd_status(settings: Settings) -> int:
    """Print one line per known broker: name | enabled | token present."""
    print(f"{'broker':<10} {'enabled':<10} {'token in keychain':<20}")
    print("-" * 42)
    for name in list_registered():
        enabled = settings.brokers_enabled.get(name.value, False)
        token = _has_token(name.value, env="production")
        print(f"{name.value:<10} {enabled!s:<10} {token!s:<20}")
    return 0


def cmd_connect(
    broker: BrokerName,
    env: str,
    *,
    prompt: Callable[[str], str] = input,
) -> int:
    """Run the one-time OAuth flow: open a URL, log in, paste the redirect back.

    No local web server is needed: after approving, the browser lands on
    the callback URL (which may show an error page — that's fine); the
    operator pastes the full address-bar URL here and the ``?code=`` in it
    is exchanged for a token, which goes to the keychain.

    The feature flag is not flipped here — it lives in ``.env`` — but the
    exact line to add is printed at the end.
    """
    from config.secrets import get_schwab_app_credentials, set_schwab_token
    from config.settings import SchwabEnv
    from ingestion.schwab_client import (
        SchwabAuthError,
        build_authorize_url,
        exchange_authorization_code,
    )

    creds = get_schwab_app_credentials(env=env)
    if creds is None:
        print(f"No Schwab app credentials in the keychain for env={env!r}.")
        print("Store your app key + secret from developer.schwab.com first:")
        print()
        print(f"    python -m config.secrets --set-schwab-app --env {env}")
        return 2

    settings = get_settings()
    redirect_uri = settings.schwab_callback_url
    url = build_authorize_url(creds.app_key, redirect_uri, SchwabEnv(env))
    print(f"Connecting broker={broker.value!r} env={env!r}.")
    print()
    print("1. Open this URL in a browser, log in to Schwab, and approve the app:")
    print()
    print(f"   {url}")
    print()
    print(f"2. You will be redirected to {redirect_uri} — the page may show an")
    print("   error; that's expected. Copy the FULL URL from the address bar.")
    print()
    redirected = prompt("3. Paste the redirect URL here> ").strip()
    code = _code_from_redirect(redirected)
    if code is None:
        print("That URL has no ?code= parameter. Nothing stored.")
        return 2

    try:
        token = exchange_authorization_code(creds, code, redirect_uri, env=SchwabEnv(env))
    except SchwabAuthError as e:
        print(f"Schwab rejected the code: {e}")
        return 2
    set_schwab_token(token, env=env)
    print()
    expires = f"{token.expires_at:%Y-%m-%d %H:%M} UTC"
    print(f"Connected. Token stored in the keychain (expires {expires};")
    print("the client refreshes it automatically for 7 days, then re-run this).")
    print()
    print("Last step — enable the broker by adding this line to quant-stack/.env:")
    print()
    print(f'    BROKERS_ENABLED=\'{{"{broker.value}": true}}\'')
    return 0


def _code_from_redirect(url: str) -> str | None:
    """Extract ``code`` from the pasted redirect URL; None if absent."""
    from urllib.parse import parse_qs, unquote, urlparse

    query = parse_qs(urlparse(url).query)
    values = query.get("code")
    if not values or not values[0]:
        return None
    return unquote(values[0])


def cmd_disconnect(broker: BrokerName, env: str) -> int:
    """Delete the token + recommend flipping the feature flag off.

    Both Schwab and TOS share the Schwab token surface in Phase 1.4a —
    same keychain entry. When TOS gets its own OAuth scope, this branches.
    """
    ok = delete_schwab_token(env=env)
    _ = broker  # branch on this when TOS diverges from Schwab token storage

    if ok:
        print(f"Deleted {broker.value} token (env={env}).")
    else:
        print(f"No {broker.value} token to delete (env={env}).")
    print(f"Remember to set brokers_enabled[{broker.value!r}] = False in your config.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _has_token(broker: str, *, env: str) -> bool:
    """True if a token exists in the keychain for this broker+env."""
    if broker in (BrokerName.schwab.value, BrokerName.tos.value):
        return get_schwab_token(env=env) is not None
    return False
