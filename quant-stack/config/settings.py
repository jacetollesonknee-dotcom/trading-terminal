"""Runtime settings for the quant engine.

Env selection is driven by ``APP_ENV`` (one of ``dev``, ``paper``, ``live``).
Non-sensitive values are read from ``.env``; sensitive values (tokens, API keys)
live in the OS keychain via :mod:`config.secrets` and are NEVER loaded here.

Live trading requires both ``APP_ENV=live`` and the boot flag
``--i-mean-it``; absent the flag, :attr:`Settings.live_orders_armed` stays
``False`` and the order router refuses to submit.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    """Runtime environment.

    - ``dev``: all writes scoped to ``data/dev/``; no Schwab calls.
    - ``paper``: real market data, simulated fills, no live order routing.
    - ``live``: live orders **only** if :attr:`Settings.live_orders_armed`.
    """

    dev = "dev"
    paper = "paper"
    live = "live"


class SchwabEnv(StrEnum):
    """Which Schwab API surface to point at."""

    production = "production"
    sandbox = "sandbox"


class BrokerName(StrEnum):
    """Brokers the registry knows how to route to.

    Both ride on the Schwab Trader API since the Ameritrade consolidation;
    the difference is account scope and ID format, not the endpoint surface.
    """

    schwab = "schwab"
    tos = "tos"


class Settings(BaseSettings):
    """Process-wide runtime configuration.

    Loaded once at startup. Never mutated. Holds only non-sensitive values;
    secrets are resolved at use-site through :mod:`config.secrets`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
    )

    # ── Runtime mode ────────────────────────────────────────────────────
    app_env: AppEnv = Field(
        default=AppEnv.dev,
        description="Runtime environment. Defaults to dev for safety.",
    )
    schwab_env: SchwabEnv = Field(
        default=SchwabEnv.sandbox,
        description="Which Schwab API surface to call. Defaults to sandbox.",
    )

    # ── Trading safety ──────────────────────────────────────────────────
    paper_trading: bool = Field(
        default=True,
        description=(
            "Hard switch. When True, every order route is intercepted by the "
            "paper trader regardless of app_env. Set to False only when "
            "app_env=live AND the boot flag --i-mean-it is set."
        ),
    )

    # ── Paths ───────────────────────────────────────────────────────────
    data_dir: Annotated[Path, Field(description="Root of the parquet/duckdb store.")] = Path(
        "data"
    )
    audit_dir: Annotated[Path, Field(description="Append-only audit JSONL files.")] = Path("audit")
    log_dir: Annotated[Path, Field(description="Structlog file sink directory.")] = Path("logs")
    memory_dir: Annotated[Path, Field(description="Structured + episodic memory.")] = Path(
        "memory"
    )

    # ── Schwab client ───────────────────────────────────────────────────
    schwab_callback_url: str = Field(
        default="https://127.0.0.1:8182",
        description="OAuth callback URL registered with Schwab developer portal.",
    )
    schwab_request_timeout_s: float = Field(default=10.0, gt=0)
    schwab_rate_limit_rps: float = Field(
        default=2.0,
        gt=0,
        description="Conservative outbound rate limit. Schwab tightens silently.",
    )

    # ── Brokers feature gate ─────────────────────────────────────────────
    brokers_enabled: dict[str, bool] = Field(
        default_factory=lambda: {b.value: False for b in BrokerName},
        description=(
            "Per-broker feature flag. False = broker registry refuses to "
            "instantiate even if a token exists in the keychain. Default "
            "is all-disabled until the operator runs `python -m cli connect <broker>`."
        ),
    )

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_json_to_file: bool = Field(default=True)

    # ─────────────────────────────────────────────────────────────────────
    #  Validators
    # ─────────────────────────────────────────────────────────────────────

    @field_validator("app_env")
    @classmethod
    def _validate_live(cls, v: AppEnv) -> AppEnv:
        """Live env is allowed only when started with --i-mean-it.

        We can't see ``paper_trading`` here (model not yet built); the
        defensive ``live_orders_armed`` computed field is the actual gate.
        """
        if v is AppEnv.live and "--i-mean-it" not in sys.argv:
            # Soft warning; the order router does the hard refusal.
            # We don't crash so dev/paper code paths can be exercised
            # against an APP_ENV=live shell without arming orders.
            pass
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def live_orders_armed(self) -> bool:
        """True only if every guardrail aligns: app_env=live, not paper_trading,
        and the boot flag was set explicitly. Order router checks this.
        """
        return (
            self.app_env is AppEnv.live
            and not self.paper_trading
            and "--i-mean-it" in sys.argv
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def scoped_data_dir(self) -> Path:
        """``data/`` namespaced by env so dev/paper/live never overlap."""
        return self.data_dir / self.app_env.value


def get_settings() -> Settings:
    """Construct settings from env + .env. Cheap; safe to call repeatedly."""
    return Settings()
