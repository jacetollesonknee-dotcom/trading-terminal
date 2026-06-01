"""Live Zacks smoke test — hits zacks.com.

Marked @pytest.mark.live; skipped by default.

KNOWN LIMITATION: Zacks renders the rank chip via JavaScript on some pages,
so server-rendered HTML can come back without rank/rank_text populated.
These tests assert only that the request succeeded and the parser returned
a valid AnalystRating envelope — not that every field is filled. A
follow-up batch (Phase 1.4c.1) will add a Playwright-based path for
JS-rendered cases when the operator's risk-of-detection rules allow.
"""

from __future__ import annotations

import pytest

from ingestion.schema import AnalystRating
from ingestion.zacks_client import ZacksClient

pytestmark = pytest.mark.live


def test_get_rating_for_nvda_returns_record() -> None:
    """Smoke: parser returns an AnalystRating envelope for NVDA."""
    with ZacksClient() as zc:
        rating = zc.get_rating("NVDA")
    assert isinstance(rating, AnalystRating)
    assert rating.symbol == "NVDA"
    assert rating.source == "zacks"


def test_get_rating_for_aapl_returns_record() -> None:
    with ZacksClient() as zc:
        rating = zc.get_rating("AAPL")
    assert isinstance(rating, AnalystRating)
    assert rating.symbol == "AAPL"
    assert rating.source == "zacks"
