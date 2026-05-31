"""`python -m cli connect ...` — manage broker connections.

Three subcommands:

    connect status                 List brokers and their connection state.
    connect <broker> [--env ENV]   Begin OAuth flow + flip the feature flag.
    connect <broker> --disconnect  Delete token + flip flag back off.

The OAuth flow itself is not wired in this phase — it raises a clear
``NotImplementedError`` with instructions, because the operator is still
provisioning Schwab API credentials. The CLI surface, the status
output, the disconnect path, and the gating are all functional.
"""

from __future__ import annotations

import argparse
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


def cmd_connect(broker: BrokerName, env: str) -> int:
    """Begin OAuth flow for ``broker``. NotImplementedError until keys land.

    The flag-flip happens *after* a successful flow. We don't enable a
    broker that can't actually be talked to.
    """
    print(f"Beginning OAuth flow for broker={broker.value!r} env={env!r}...")
    print()
    print("  Phase 1.4a: stub. The OAuth flow lands when you have your new")
    print("  Schwab API credentials provisioned and the callback URL")
    print("  registered. For now, when your keys are ready, seed the token")
    print("  manually with:")
    print()
    print(f"    python -m config.secrets --set-schwab-token --env {env}")
    print()
    print("  Then flip the flag in your .env or settings:")
    print(f"    brokers_enabled['{broker.value}'] = True")
    print()
    print("  Future work in this same command will wire the full OAuth flow.")
    return 2  # advisory — not a hard error, but didn't accomplish a connect


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
