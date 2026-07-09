"""Unit tests for the terminal's engine bridge.

``terminal/`` is excluded from the engine's strict gates (see pyproject), but
``engine_bridge.py`` carries real field-mapping and graceful-degradation logic
worth locking down. We put ``terminal/`` on ``sys.path`` before importing it so
the bridge runs inside the normal engine pytest run without needing Flask.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ingestion.schema import AnalystRating, InsiderTrade
from storage.store import ParquetStore

_TERMINAL = Path(__file__).resolve().parents[2] / "terminal"
if str(_TERMINAL) not in sys.path:
    sys.path.insert(0, str(_TERMINAL))

# Imported after the sys.path insert above so the terminal package resolves.
from engine_bridge import (  # noqa: E402  (import-after-path-setup is intentional)
    EngineBridge,
    _insider_to_row,
    _rating_to_zacks,
)

_NOW = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


# ── pure mappers ─────────────────────────────────────────────────────────


def test_insider_to_row_maps_engine_fields_to_terminal_keys() -> None:
    trade = InsiderTrade(
        as_of=_NOW,
        filing_date=date(2026, 7, 8),
        trade_date=date(2026, 7, 7),
        ticker="NVDA",
        company="NVIDIA",
        insider_name="Jensen Huang",
        insider_title="CEO",
        trade_type="P - Purchase",
        price_per_share=120.5,
        quantity=1000,
        shares_owned_after=50000,
        dollar_value=120500.0,
        source="openinsider",
    )
    row = _insider_to_row(trade)
    # Terminal table contract: title/price/value, not insider_title/price_per_share/dollar_value.
    assert row["title"] == "CEO"
    assert row["price"] == 120.5
    assert row["value"] == 120500.0
    assert row["trade_date"] == "2026-07-07"
    assert row["ticker"] == "NVDA"
    assert row["source"] == "engine"


def test_insider_to_row_blanks_none_numerics() -> None:
    trade = InsiderTrade(
        as_of=_NOW,
        filing_date=date(2026, 7, 8),
        trade_date=date(2026, 7, 7),
        ticker="AAPL",
        insider_name="Someone",
        trade_type="M - Option Exercise",
        price_per_share=None,
        quantity=10,
        shares_owned_after=None,
        dollar_value=None,
        source="openinsider",
    )
    row = _insider_to_row(trade)
    # None → "" so the front-end renders an empty cell rather than "None".
    assert row["price"] == ""
    assert row["owned"] == ""
    assert row["value"] == ""


def test_rating_to_zacks_shapes_card_and_builds_stats() -> None:
    rating = AnalystRating(
        as_of=_NOW,
        symbol="NVDA",
        rank=2,
        rank_text="Buy",
        price_target=175.0,
        style_score_value="C",
        style_score_growth="A",
        style_score_momentum="B",
        style_score_vgm="B",
        industry_rank_text="Top 12%",
        source="zacks",
    )
    card = _rating_to_zacks(rating)
    assert card["zacks_rank"] == "2"
    assert card["rank_text"] == "Buy"
    assert card["price_target"] == "$175.00"
    assert card["stats"] == {
        "Value": "C",
        "Growth": "A",
        "Momentum": "B",
        "VGM": "B",
        "Industry Rank": "Top 12%",
    }


def test_rating_to_zacks_handles_missing_fields() -> None:
    rating = AnalystRating(as_of=_NOW, symbol="SPY", source="zacks")
    card = _rating_to_zacks(rating)
    assert card["zacks_rank"] == "N/A"
    assert card["price_target"] is None
    assert card["stats"] == {}


# ── graceful degradation ─────────────────────────────────────────────────


def test_disabled_bridge_is_unavailable_and_inert() -> None:
    b = EngineBridge(enabled=False)
    assert b.available is False
    assert b.start(lambda *_a: None) is False
    assert b.sentiment("NVDA") == []
    assert b.insider("NVDA") == []
    assert b.zacks("NVDA") is None
    # set_watchlist / stop must not raise when there's no scheduler.
    b.set_watchlist(["NVDA"])
    b.stop()
    st = b.status()
    assert st["enabled"] is False
    assert st["available"] is False
    assert st["running"] is False


# ── end-to-end read-through a real store ─────────────────────────────────


@pytest.fixture
def dev_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParquetStore:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_ENV", "dev")
    return ParquetStore(tmp_path, "dev")


def test_bridge_reads_insider_and_zacks_through_store(dev_store: ParquetStore) -> None:
    dev_store.write_insider_trades([
        InsiderTrade(
            as_of=_NOW,
            filing_date=date(2026, 7, 8),
            trade_date=date(2026, 7, 7),
            ticker="NVDA",
            insider_name="Jensen Huang",
            insider_title="CEO",
            trade_type="P - Purchase",
            price_per_share=120.5,
            quantity=1000,
            source="openinsider",
        )
    ])
    dev_store.write_analyst_ratings([
        AnalystRating(as_of=_NOW, symbol="NVDA", rank=1, rank_text="Strong Buy", source="zacks")
    ])

    b = EngineBridge(enabled=True)
    assert b.available is True

    rows = b.insider("NVDA")
    assert len(rows) == 1
    assert rows[0]["insider_name"] == "Jensen Huang"
    assert rows[0]["source"] == "engine"

    card = b.zacks("NVDA")
    assert card is not None
    assert card["zacks_rank"] == "1"
    assert card["rank_text"] == "Strong Buy"

    # Unknown symbol → empty/None, never an exception.
    assert b.insider("ZZZZ") == []
    assert b.zacks("ZZZZ") is None
