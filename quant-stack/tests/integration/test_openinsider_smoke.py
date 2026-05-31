"""Live OpenInsider smoke test — hits openinsider.com.

Marked @pytest.mark.live; skipped by default. Run with:
    uv run pytest -m live tests/integration/test_openinsider_smoke.py
"""

from __future__ import annotations

import pytest

from ingestion.openinsider_client import OpenInsiderClient

pytestmark = pytest.mark.live


def test_latest_trades_returns_at_least_one() -> None:
    with OpenInsiderClient() as oi:
        trades = oi.get_latest_trades()
    assert len(trades) >= 1
    t = trades[0]
    assert t.ticker
    assert t.insider_name
    assert t.trade_type
    assert t.source == "openinsider"


def test_top_purchases_of_month_returns_at_least_one() -> None:
    with OpenInsiderClient() as oi:
        trades = oi.get_top_purchases_of_month()
    assert len(trades) >= 1
    # By construction this page only shows purchases.
    assert any("Purchase" in t.trade_type for t in trades)


def test_per_symbol_query_for_nvda_works() -> None:
    """NVDA is large-cap; almost always has at least one Form-4 in 30 days."""
    with OpenInsiderClient() as oi:
        trades = oi.get_trades_for_symbol("NVDA", lookback_days=90)
    # Don't assert >=1 — a quiet quarter is possible. Just assert call shape.
    assert isinstance(trades, list)
    for t in trades:
        assert t.ticker == "NVDA"
