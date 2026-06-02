"""Live X (rsshub.app) smoke test.

Marked @pytest.mark.live; skipped by default.

rsshub.app is community-run and not 100% reliable. These tests are
deliberately permissive: a successful request and a well-formed
list[SocialPost] envelope is the bar — content can be empty if rsshub
is rate-limited upstream from X.
"""

from __future__ import annotations

import pytest

from ingestion.schema import SocialPost
from ingestion.x_client import XClient, XError

pytestmark = pytest.mark.live


# A few stable, high-volume X handles to spot-check. If rsshub is down
# every test below will skip — not fail — so the suite stays usable.
_TEST_HANDLES = ("CNBCnow", "MarketWatch")


def test_get_posts_for_handle_returns_list() -> None:
    """At least one of the test handles should return a non-empty feed.

    If rsshub.app or its upstream connection to X is down, we skip
    rather than fail — same intent as the OpenInsider smoke test.
    """
    last_err: Exception | None = None
    for handle in _TEST_HANDLES:
        try:
            with XClient() as xc:
                posts = xc.get_posts_for_handle(handle)
        except XError as e:
            last_err = e
            continue
        assert isinstance(posts, list)
        for p in posts:
            assert isinstance(p, SocialPost)
            assert p.platform == "x"
            assert p.source == "rsshub"
        if posts:
            return  # Found a live handle; pass.
    pytest.skip(f"rsshub.app unreachable or all handles empty (last error: {last_err})")
