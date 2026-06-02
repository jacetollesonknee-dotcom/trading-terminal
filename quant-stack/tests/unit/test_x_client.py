"""X (Twitter) client tests — respx-mocked RSS fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Final

import httpx
import pytest
import respx

from ingestion.schema import SocialPost
from ingestion.x_client import XClient, XError, _parse_feed
from storage.query import PointInTimeQuery
from storage.store import ParquetStore

_BASE: Final[str] = "https://rsshub.app"


def _feed(items_xml: str, *, channel_title: str = "Jane Trader") -> str:
    return dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>{channel_title}</title>
            <link>https://x.com/janetrader</link>
            <description>Recent posts by janetrader</description>
            {items_xml}
          </channel>
        </rss>
        """
    )


_ITEM_PLAIN = dedent("""\
<item>
  <title>Watching $NVDA after the close. Big setup.</title>
  <description>Watching $NVDA after the close. Big setup.</description>
  <link>https://x.com/janetrader/status/12345</link>
  <guid>https://x.com/janetrader/status/12345</guid>
  <pubDate>Sun, 31 May 2026 14:30:00 GMT</pubDate>
  <author>janetrader</author>
</item>""")

_ITEM_WITH_MENTIONS = dedent("""\
<item>
  <title>great call by @bullishbob on $TSLA — also love $NVDA here</title>
  <description>great call by @bullishbob on $TSLA — also love $NVDA here</description>
  <link>https://x.com/janetrader/status/12346</link>
  <guid>https://x.com/janetrader/status/12346</guid>
  <pubDate>Mon, 01 Jun 2026 09:15:00 GMT</pubDate>
  <author>janetrader</author>
</item>""")

_ITEM_RETWEET = dedent("""\
<item>
  <title>RT @bullishbob: $AAPL printing money this week</title>
  <description>RT @bullishbob: $AAPL printing money this week</description>
  <link>https://x.com/janetrader/status/12347</link>
  <guid>https://x.com/janetrader/status/12347</guid>
  <pubDate>Mon, 01 Jun 2026 10:00:00 GMT</pubDate>
  <author>janetrader</author>
</item>""")

_ITEM_NO_PUBDATE = dedent("""\
<item>
  <title>no pubdate here</title>
  <description>generic post</description>
  <link>https://x.com/janetrader/status/12348</link>
  <guid>https://x.com/janetrader/status/12348</guid>
  <author>janetrader</author>
</item>""")

_ITEM_MALFORMED = "<item><title>broken</title></item>"  # no guid, no link


# ─────────────────────────────────────────────────────────────────────────
#  Parser direct tests
# ─────────────────────────────────────────────────────────────────────────


def test_parse_plain_post() -> None:
    posts = _parse_feed(_feed(_ITEM_PLAIN), fallback_handle="janetrader",
                        as_of=datetime.now(UTC))
    assert len(posts) == 1
    p = posts[0]
    assert p.platform == "x"
    assert p.author_handle == "janetrader"
    assert "$NVDA" in p.content
    assert p.cashtags == ("NVDA",)
    assert p.url == "https://x.com/janetrader/status/12345"
    assert p.posted_at is not None
    assert p.posted_at.year == 2026
    assert p.posted_at.tzinfo is not None
    assert p.reply_to is None
    assert p.source == "rsshub"


def test_parse_multi_cashtag_with_mention() -> None:
    posts = _parse_feed(_feed(_ITEM_WITH_MENTIONS), fallback_handle="janetrader",
                        as_of=datetime.now(UTC))
    p = posts[0]
    assert set(p.cashtags) == {"TSLA", "NVDA"}
    assert "bullishbob" in p.mentions


def test_parse_retweet_sets_reply_to() -> None:
    posts = _parse_feed(_feed(_ITEM_RETWEET), fallback_handle="janetrader",
                        as_of=datetime.now(UTC))
    p = posts[0]
    assert p.reply_to == "bullishbob"
    assert "AAPL" in p.cashtags


def test_parse_item_with_no_pubdate_keeps_posted_at_none() -> None:
    posts = _parse_feed(_feed(_ITEM_NO_PUBDATE), fallback_handle="janetrader",
                        as_of=datetime.now(UTC))
    assert posts[0].posted_at is None


def test_parser_skips_malformed_items() -> None:
    posts = _parse_feed(
        _feed(_ITEM_PLAIN + _ITEM_MALFORMED + _ITEM_WITH_MENTIONS),
        fallback_handle="janetrader",
        as_of=datetime.now(UTC),
    )
    assert len(posts) == 2


def test_parse_empty_channel() -> None:
    posts = _parse_feed(_feed(""), fallback_handle="x", as_of=datetime.now(UTC))
    assert posts == []


def test_parser_strips_html_entities_and_tags() -> None:
    item = dedent("""\
    <item>
      <title>price &amp; vol surge for $NVDA</title>
      <description>&lt;p&gt;price &amp;amp; vol surge for $NVDA&lt;/p&gt;</description>
      <link>https://x.com/janetrader/status/77</link>
      <guid>https://x.com/janetrader/status/77</guid>
      <pubDate>Sun, 31 May 2026 14:30:00 GMT</pubDate>
      <author>janetrader</author>
    </item>""")
    posts = _parse_feed(_feed(item), fallback_handle="janetrader", as_of=datetime.now(UTC))
    p = posts[0]
    assert "<p>" not in p.content
    assert "$NVDA" in p.content


# ─────────────────────────────────────────────────────────────────────────
#  Client request paths
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_get_posts_for_handle_round_trip() -> None:
    respx.get(f"{_BASE}/twitter/user/janetrader").mock(
        return_value=httpx.Response(200, text=_feed(_ITEM_PLAIN + _ITEM_WITH_MENTIONS))
    )
    with XClient() as xc:
        posts = xc.get_posts_for_handle("janetrader")
    assert len(posts) == 2


@respx.mock
def test_get_posts_strips_leading_at() -> None:
    respx.get(f"{_BASE}/twitter/user/janetrader").mock(
        return_value=httpx.Response(200, text=_feed(_ITEM_PLAIN))
    )
    with XClient() as xc:
        posts = xc.get_posts_for_handle("@janetrader")
    assert len(posts) == 1


def test_empty_handle_raises_value_error() -> None:
    with XClient() as xc:  # noqa: SIM117
        with pytest.raises(ValueError, match="non-empty"):
            xc.get_posts_for_handle("@")


@respx.mock
def test_network_failure_wraps_in_x_error() -> None:
    respx.get(f"{_BASE}/twitter/user/janetrader").mock(
        side_effect=httpx.TransportError("rsshub down")
    )
    with XClient() as xc:  # noqa: SIM117
        with pytest.raises(XError, match="failed"):
            xc.get_posts_for_handle("janetrader")


@respx.mock
def test_get_posts_for_handles_skips_one_bad_handle() -> None:
    respx.get(f"{_BASE}/twitter/user/good").mock(
        return_value=httpx.Response(200, text=_feed(_ITEM_PLAIN))
    )
    respx.get(f"{_BASE}/twitter/user/bad").mock(
        side_effect=httpx.TransportError("boom")
    )
    with XClient() as xc:
        posts = xc.get_posts_for_handles(["good", "bad"])
    assert len(posts) == 1


@respx.mock
def test_custom_base_url_used() -> None:
    custom = "http://localhost:1200"
    respx.get(f"{custom}/twitter/user/janetrader").mock(
        return_value=httpx.Response(200, text=_feed(_ITEM_PLAIN))
    )
    with XClient(base_url=custom) as xc:
        posts = xc.get_posts_for_handle("janetrader")
    assert len(posts) == 1


# ─────────────────────────────────────────────────────────────────────────
#  Storage round trip
# ─────────────────────────────────────────────────────────────────────────


@respx.mock
def test_storage_round_trip(tmp_path: Path) -> None:
    respx.get(f"{_BASE}/twitter/user/janetrader").mock(
        return_value=httpx.Response(200, text=_feed(_ITEM_PLAIN + _ITEM_WITH_MENTIONS))
    )
    store = ParquetStore(tmp_path, env="test")
    with XClient() as xc:
        posts = xc.get_posts_for_handle("janetrader")
    result = store.write_social_posts(posts)
    assert result.persisted == 2

    q = PointInTimeQuery(store, as_of=datetime.now(UTC))
    got = q.social_posts("janetrader")
    assert len(got) == 2
    # Newest first
    assert got[0].posted_at is not None
    assert got[1].posted_at is not None
    assert got[0].posted_at >= got[1].posted_at


@respx.mock
def test_storage_dedup_across_polls(tmp_path: Path) -> None:
    respx.get(f"{_BASE}/twitter/user/janetrader").mock(
        return_value=httpx.Response(200, text=_feed(_ITEM_PLAIN))
    )
    store = ParquetStore(tmp_path, env="test")
    with XClient() as xc:
        r1 = store.write_social_posts(xc.get_posts_for_handle("janetrader"))
        r2 = store.write_social_posts(xc.get_posts_for_handle("janetrader"))
    assert r1.persisted == 1
    assert r2.persisted == 0
    assert r2.deduplicated == 1


@respx.mock
def test_query_filter_by_cashtag(tmp_path: Path) -> None:
    respx.get(f"{_BASE}/twitter/user/janetrader").mock(
        return_value=httpx.Response(
            200, text=_feed(_ITEM_PLAIN + _ITEM_WITH_MENTIONS + _ITEM_RETWEET)
        )
    )
    store = ParquetStore(tmp_path, env="test")
    with XClient() as xc:
        store.write_social_posts(xc.get_posts_for_handle("janetrader"))

    q = PointInTimeQuery(store, as_of=datetime.now(UTC))
    nvda_posts = q.social_posts("janetrader", cashtag="NVDA")
    assert len(nvda_posts) == 2
    aapl_posts = q.social_posts("janetrader", cashtag="AAPL")
    assert len(aapl_posts) == 1


@respx.mock
def test_cross_author_mention_search(tmp_path: Path) -> None:
    """social_posts_mentioning() reaches across all authors on the platform."""
    respx.get(f"{_BASE}/twitter/user/alice").mock(
        return_value=httpx.Response(200, text=_feed(_ITEM_PLAIN,
                                                    channel_title="Alice"))
    )
    respx.get(f"{_BASE}/twitter/user/bob").mock(
        return_value=httpx.Response(200, text=_feed(_ITEM_WITH_MENTIONS,
                                                    channel_title="Bob"))
    )
    store = ParquetStore(tmp_path, env="test")
    with XClient() as xc:
        store.write_social_posts(xc.get_posts_for_handle("alice"))
        store.write_social_posts(xc.get_posts_for_handle("bob"))

    q = PointInTimeQuery(store, as_of=datetime.now(UTC))
    nvda_mentions = q.social_posts_mentioning("NVDA")
    assert len(nvda_mentions) == 2


def test_decision_time_filters_future_posts(tmp_path: Path) -> None:
    """Posts with as_of after decision_time must be invisible."""
    store = ParquetStore(tmp_path, env="test")
    past = SocialPost(
        as_of=datetime(2024, 1, 15, 14, 0, tzinfo=UTC),
        platform="x", post_id="p1",
        author_handle="janetrader", content="$NVDA",
        posted_at=datetime(2024, 1, 15, 13, 0, tzinfo=UTC),
        cashtags=("NVDA",), source="rsshub",
    )
    future = SocialPost(
        as_of=datetime(2024, 3, 1, 14, 0, tzinfo=UTC),
        platform="x", post_id="p2",
        author_handle="janetrader", content="$TSLA",
        posted_at=datetime(2024, 3, 1, 13, 0, tzinfo=UTC),
        cashtags=("TSLA",), source="rsshub",
    )
    store.write_social_posts([past, future])

    q = PointInTimeQuery(store, as_of=datetime(2024, 2, 1, tzinfo=UTC))
    got = q.social_posts("janetrader")
    assert len(got) == 1
    assert got[0].post_id == "p1"
