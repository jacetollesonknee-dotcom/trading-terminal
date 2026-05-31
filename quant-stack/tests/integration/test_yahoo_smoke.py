"""Live Yahoo smoke test — Phase 0 §5 deliverable, now reusable.

Pulls a small slice of real Yahoo data, persists via the storage layer,
and reads it back through PointInTimeQuery. Asserts shape, not values.

Marked ``@pytest.mark.live``. Skipped by default. To run:

    uv run pytest -m live tests/integration/test_yahoo_smoke.py
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingestion.yahoo_client import YahooClient
from storage.query import PointInTimeQuery
from storage.store import ParquetStore

pytestmark = pytest.mark.live


def test_yahoo_round_trip_spy_5d_1d() -> None:
    """Pull 5 days of SPY 1d bars, persist, read back via point-in-time query."""
    with YahooClient() as yc:
        bars = yc.get_history("SPY", period="5d", interval="1d")

    assert len(bars) >= 3, f"expected at least 3 bars; got {len(bars)}"
    assert all(b.symbol == "SPY" for b in bars)
    assert all(b.source == "yahoo" for b in bars)
    assert all(b.as_of.tzinfo is not None for b in bars)

    with tempfile.TemporaryDirectory() as td:
        store = ParquetStore(Path(td), env="smoke")
        result = store.write_equity_bars(bars)
        assert result.persisted == len(bars)
        assert result.deduplicated == 0

        q = PointInTimeQuery(store, as_of=datetime.now(UTC))
        got = q.equity_bars("SPY")
        assert len(got) == len(bars)
        assert got[-1].close == bars[-1].close


def test_yahoo_nvda_splits_visible() -> None:
    """NVDA had a 10:1 split on 2024-06-10. It must be in the 10-year window."""
    with YahooClient() as yc:
        actions = yc.get_splits_and_dividends("NVDA", period="10y")

    splits = [a for a in actions if a.type == "split"]
    assert splits, "expected at least one NVDA split in the 10-year window"
    # The 2024-06-10 10:1 split (date or close-to-it). Approximate match.
    ratios = {a.ratio for a in splits}
    assert any(r is not None and r >= 4.0 for r in ratios), (
        f"expected at least one split with ratio>=4; got ratios={ratios}"
    )


def test_yahoo_dividend_amount_is_positive_for_kmi() -> None:
    """KMI pays a regular cash dividend. Sanity-check at least one is positive."""
    with YahooClient() as yc:
        actions = yc.get_splits_and_dividends("KMI", period="2y")
    divs = [a for a in actions if a.type == "cash_dividend"]
    assert divs, "expected at least one KMI dividend in the 2-year window"
    assert all(a.cash_amount is not None and a.cash_amount > 0 for a in divs)
