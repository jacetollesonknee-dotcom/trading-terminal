"""Settings tests — primarily the live-orders-armed guardrail."""

from __future__ import annotations

import sys
from unittest.mock import patch

from config.settings import AppEnv, SchwabEnv, Settings


def test_defaults_are_safe() -> None:
    """Default construction is dev / sandbox / paper-on."""
    s = Settings()
    assert s.app_env is AppEnv.dev
    assert s.schwab_env is SchwabEnv.sandbox
    assert s.paper_trading is True
    assert s.live_orders_armed is False


def test_live_orders_armed_requires_all_three() -> None:
    """app_env=live alone is not enough. Need paper_trading=False AND --i-mean-it."""
    # live + paper_trading=False but no boot flag → still disarmed
    with patch.object(sys, "argv", ["pytest"]):
        s = Settings(app_env=AppEnv.live, paper_trading=False)
        assert s.live_orders_armed is False

    # live + paper_trading=False + boot flag → armed
    with patch.object(sys, "argv", ["pytest", "--i-mean-it"]):
        s = Settings(app_env=AppEnv.live, paper_trading=False)
        assert s.live_orders_armed is True

    # live + paper_trading=True + boot flag → still disarmed (paper wins)
    with patch.object(sys, "argv", ["pytest", "--i-mean-it"]):
        s = Settings(app_env=AppEnv.live, paper_trading=True)
        assert s.live_orders_armed is False

    # paper env + boot flag → disarmed regardless
    with patch.object(sys, "argv", ["pytest", "--i-mean-it"]):
        s = Settings(app_env=AppEnv.paper, paper_trading=False)
        assert s.live_orders_armed is False


def test_scoped_data_dir_isolates_by_env() -> None:
    """dev/paper/live writes never overlap."""
    assert Settings(app_env=AppEnv.dev).scoped_data_dir.name == "dev"
    assert Settings(app_env=AppEnv.paper).scoped_data_dir.name == "paper"
    assert Settings(app_env=AppEnv.live).scoped_data_dir.name == "live"


def test_settings_is_frozen() -> None:
    """Settings cannot be mutated after construction."""
    s = Settings()
    try:
        s.app_env = AppEnv.live  # type: ignore[misc]
    except (TypeError, ValueError):
        return
    msg = "Settings should be frozen"
    raise AssertionError(msg)
