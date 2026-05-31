"""Entry point: ``python -m cli <command> [...]``.

Subcommands:
    connect    Manage broker connections (Schwab, TOS).
"""

from __future__ import annotations

import argparse

from cli import connect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cli",
        description="Quant-stack engine operator CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    connect.add_subparser(subparsers)

    args = parser.parse_args(argv)
    if args.command == "connect":
        return connect.dispatch(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
