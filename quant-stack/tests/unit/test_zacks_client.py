"""Zacks client tests — respx-mocked HTML fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Final

import httpx
import pytest
import respx

from ingestion.schema import AnalystRating
from ingestion.zacks_client import ZacksClient, ZacksError, _parse_rating
from storage.query import PointInTimeQuery
from storage.store import ParquetStore

_BASE: Final[str] = "https://www.zacks.com"


_FULL_PAGE = dedent("""\
<html><body>
<div class="zr_rankbox">
  <span class="rank_view">1</span>
  <span class="rank_chip">Strong Buy</span>
</div>
<div id="price_target_summary">$210.50</div>
<div class="composite_val_rank">Value: A</div>
<div class="composite_val_rank">Growth: B</div>
<div class="composite_val_rank">Momentum: A</div>
<div class="composite_val_rank">VGM: A</div>
<div class="sector_industry_rank">25 / 250 (Top 10%)</div>
</body></html>
""")

_MINIMAL_PAGE = dedent("""\
<html><body>
<div class="zr_rankbox"><span class="rank_chip">Hold</span></div>
</body></html>
""")

_EMPTY_PAGE = "<html><body><p>No data</p></body></html>"


# ─────────────────────────────────────────────────────────────────────────
#  Parser direct tests
# ─────────────────────────────────────────────────────────────────────────


def test_parse_full_page() -> None:
    r = _parse_rating(_FULL_PAGE, symbol="NVDA", as_of=datetime.now(UTC))
    assert r.symbol == "NVDA"
    assert r.rank == 1
    assert r.rank_text == "Strong Buy"
    assert r.price_target == 210.50
    assert r.style_score_value == "A"
    assert r.style_score_growth == "B"
    assert r.style_score_momentum == "A"
    assert r.style_score_vgm == "A"
    assert r.industry_rank == 25
    assert "25 / 250" in (r.industry_rank_text or "")
    assert r.source == "zacks"


def test_parse_minimal_page_backfills_rank_num_from_text() -> None:
    r = _parse_rating(_MINIMAL_PAGE, symbol="X", as_of=datetime.now(UTC))
    assert r.rank == 3
    assert r.rank_text == "Hold"
    assert r.price_target is None
    assert r.style_score_value is None


def test_parse_empty_page_returns_mostly_nones() -> None:
    r = _parse_rating(_EMPTY_PAGE, symbol="X", as_of=datetime.now(UTC))
    assert r.symbol == "X"
    assert r.rank is None
    assert r.rank_text is None
    assert r.price_target is None
    assert r.source == "zacks"


def test_parse_handles_commas_in_price_target() -> None:
    html = '<html><body><div id="price_target_summary">$1,234.56</div></body></html>'
    r = _parse_rating(html, symbol="X", as_of=datetime.now(UTC))
    assert r.price_target == 1234.56


# ─────────────────────────────────────────────────────────────────────────
#  Client request paths
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_get_rating_round_trip() -> None:
    respx.get(f"{_BASE}/stock/quote/NVDA").mock(
        return_value=httpx.Response(200, text=_FULL_PAGE)
    )
    with ZacksClient() as zc:
        rating = zc.get_rating("nvda")
    assert rating.symbol == "NVDA"
    assert rating.rank == 1


@respx.mock
def test_network_failure_wraps_in_zacks_error() -> None:
    respx.get(f"{_BASE}/stock/quote/NVDA").mock(
        side_effect=httpx.TransportError("boom")
    )
    with ZacksClient() as zc:  # noqa: SIM117
        with pytest.raises(ZacksError, match="failed"):
            zc.get_rating("NVDA")


@respx.mock
def test_404_raises_zacks_error() -> None:
    respx.get(f"{_BASE}/stock/quote/FAKE").mock(
        return_value=httpx.Response(404, text="not found")
    )
    with ZacksClient() as zc:  # noqa: SIM117
        with pytest.raises(ZacksError):
            zc.get_rating("FAKE")


# ─────────────────────────────────────────────────────────────────────────
#  Storage round trip + daily dedup
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_storage_round_trip(tmp_path: Path) -> None:
    respx.get(f"{_BASE}/stock/quote/NVDA").mock(
        return_value=httpx.Response(200, text=_FULL_PAGE)
    )
    store = ParquetStore(tmp_path, env="test")
    with ZacksClient() as zc:
        rating = zc.get_rating("NVDA")
    result = store.write_analyst_ratings([rating])
    assert result.persisted == 1

    q = PointInTimeQuery(store, as_of=datetime.now(UTC))
    got = q.latest_analyst_rating("NVDA")
    assert got is not None
    assert got.rank == 1
    assert got.rank_text == "Strong Buy"
    assert got.price_target == 210.50


@respx.mock
def test_daily_dedup_same_day(tmp_path: Path) -> None:
    """Multiple polls in the same day → only one record persisted."""
    respx.get(f"{_BASE}/stock/quote/NVDA").mock(
        return_value=httpx.Response(200, text=_FULL_PAGE)
    )
    store = ParquetStore(tmp_path, env="test")
    with ZacksClient() as zc:
        r1 = zc.get_rating("NVDA")
        r2 = zc.get_rating("NVDA")
    s1 = store.write_analyst_ratings([r1])
    s2 = store.write_analyst_ratings([r2])
    assert s1.persisted == 1
    assert s2.persisted == 0
    assert s2.deduplicated == 1


def test_analyst_ratings_query_filtered_by_decision_time(tmp_path: Path) -> None:
    """A rating taken AFTER the decision_time must be invisible."""
    store = ParquetStore(tmp_path, env="test")
    past = AnalystRating(
        as_of=datetime(2024, 1, 15, 14, 0, tzinfo=UTC),
        symbol="X", rank=2, rank_text="Buy",
        source="zacks",
    )
    future = AnalystRating(
        as_of=datetime(2024, 3, 1, 14, 0, tzinfo=UTC),
        symbol="X", rank=1, rank_text="Strong Buy",
        source="zacks",
    )
    store.write_analyst_ratings([past, future])

    q = PointInTimeQuery(store, as_of=datetime(2024, 2, 1, tzinfo=UTC))
    got = q.analyst_ratings("X")
    assert len(got) == 1
    assert got[0].rank == 2  # only the past one
