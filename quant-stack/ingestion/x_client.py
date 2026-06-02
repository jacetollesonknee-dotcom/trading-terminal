"""X (Twitter) client via rsshub.app — free RSS proxy.

X's official API is paid; rsshub.app is a community-maintained proxy that
exposes X user feeds as RSS for free. The engine treats rsshub as the
source-of-record for now; if the operator runs their own rsshub instance
they can swap the base URL at construction time.

URL pattern:
    https://rsshub.app/twitter/user/{handle}

The RSS feed includes per-item guid (used as post_id), pubDate, link, and
description. The description contains the post text plus any
quote-tweet / reply context. We extract:

- post_id: from RSS <guid>
- posted_at: from <pubDate>
- content: plain-text from <description>
- url: from <link>
- cashtags: regex-extracted from content ($NVDA → 'NVDA')
- mentions: regex-extracted from content (@handle)

KNOWN LIMITATIONS:
- rsshub.app is community-run; uptime is not guaranteed. The retry helper
  treats 502/503 as transient. Build for graceful failure (caller sees an
  empty list when the proxy is down).
- Some X handles return empty feeds if rate-limited upstream. Not an error
  on our side.
- Posts older than ~30 days drop off the RSS window. For deeper history
  the operator would need the paid X API.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Final

import httpx
from bs4 import BeautifulSoup, Tag

from ingestion._http import HTTPError, get_with_retry
from ingestion.schema import SocialPost

_DEFAULT_BASE_URL: Final[str] = "https://rsshub.app"
_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (compatible; quant-stack/1.0; +https://github.com/)"
)
_DEFAULT_TIMEOUT_S: Final[float] = 15.0

# Cashtag: $TICKER where ticker is 1-5 uppercase letters (optional ".X" suffix).
_CASHTAG_RE: Final[re.Pattern[str]] = re.compile(r"\$([A-Z]{1,5}(?:\.[A-Z])?)\b")
# @handle: 1-15 alphanumeric/underscore (X username limit is 15).
_MENTION_RE: Final[re.Pattern[str]] = re.compile(r"@([A-Za-z0-9_]{1,15})")
# RT detection: "RT @other:" prefix
_RT_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^RT\s+@(\w+):", re.IGNORECASE)


class XError(RuntimeError):
    """rsshub request failed or returned an unparseable response."""


# ─────────────────────────────────────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────────────────────────────────────


class XClient:
    """Sync httpx client for rsshub.app's X user feeds.

    Lifecycle::

        with XClient() as xc:
            posts = xc.get_posts_for_handle("elonmusk")

    Run your own rsshub: pass ``base_url="http://localhost:1200"``.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/rss+xml,text/xml"},
            timeout=timeout_s,
            follow_redirects=True,
        )
        self._base_url = base_url.rstrip("/")

    def __enter__(self) -> XClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._client.close()

    # ── public API ────────────────────────────────────────────────────────

    def get_posts_for_handle(self, handle: str) -> list[SocialPost]:
        """Recent posts from one X user.

        Polite cadence: 5 minutes during market hours. rsshub caches
        upstream so polling more aggressively just gets the same feed.
        """
        handle_clean = handle.lstrip("@")
        if not handle_clean:
            msg = "handle must be non-empty"
            raise ValueError(msg)
        xml = self._get(f"/twitter/user/{handle_clean}")
        as_of = datetime.now(UTC)
        return list(_parse_feed(xml, fallback_handle=handle_clean, as_of=as_of))

    def get_posts_for_handles(self, handles: Iterable[str]) -> list[SocialPost]:
        """Convenience: pull each handle's feed, concatenate."""
        out: list[SocialPost] = []
        for h in handles:
            try:
                out.extend(self.get_posts_for_handle(h))
            except XError:
                # One bad handle shouldn't poison the whole pull.
                continue
        return out

    # ── internals ─────────────────────────────────────────────────────────

    def _get(self, path: str) -> str:
        try:
            resp = get_with_retry(self._client, path)
        except HTTPError as e:
            msg = f"X (rsshub) GET {path} failed: {e}"
            raise XError(msg) from e
        return resp.text


# ─────────────────────────────────────────────────────────────────────────────
#  Parser — pure function, exported for direct testing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_feed(
    xml: str, *, fallback_handle: str, as_of: datetime
) -> list[SocialPost]:
    """Parse an rsshub /twitter/user/<handle> feed into SocialPost records."""
    soup = BeautifulSoup(xml, features="xml")
    channel = soup.find("channel")
    if not isinstance(channel, Tag):
        return []

    # Channel-level author name
    channel_title_el = channel.find("title")
    channel_title = (
        channel_title_el.get_text(strip=True) if isinstance(channel_title_el, Tag) else None
    )

    posts: list[SocialPost] = []
    for item in channel.find_all("item"):
        try:
            posts.append(
                _item_to_post(
                    item,
                    fallback_handle=fallback_handle,
                    channel_title=channel_title,
                    as_of=as_of,
                )
            )
        except (ValueError, AttributeError, TypeError):
            # rsshub occasionally emits a malformed item; skip silently.
            continue
    return posts


def _item_to_post(
    item: Tag,
    *,
    fallback_handle: str,
    channel_title: str | None,
    as_of: datetime,
) -> SocialPost:
    guid_el = item.find("guid")
    link_el = item.find("link")
    pub_el = item.find("pubDate")
    desc_el = item.find("description")
    title_el = item.find("title")
    author_el = item.find("author") or item.find("dc:creator")

    post_id = (
        (guid_el.get_text(strip=True) if isinstance(guid_el, Tag) else None)
        or (link_el.get_text(strip=True) if isinstance(link_el, Tag) else None)
        or ""
    )
    if not post_id:
        msg = "item has neither guid nor link"
        raise ValueError(msg)

    url = link_el.get_text(strip=True) if isinstance(link_el, Tag) else None

    # Posted-at parsing — RFC 2822 from pubDate
    posted_at: datetime | None = None
    if isinstance(pub_el, Tag):
        try:
            posted_at = parsedate_to_datetime(pub_el.get_text(strip=True))
            if posted_at and posted_at.tzinfo is None:
                posted_at = posted_at.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            posted_at = None

    # Content: prefer description (richer), fall back to title.
    raw_html = (
        (desc_el.get_text() if isinstance(desc_el, Tag) else None)
        or (title_el.get_text() if isinstance(title_el, Tag) else None)
        or ""
    )
    content = _strip_html(raw_html).strip()

    # Reply-to: detect "RT @other:" prefix as a quote/retweet marker.
    reply_to: str | None = None
    rt_match = _RT_PREFIX_RE.match(content)
    if rt_match:
        reply_to = rt_match.group(1)

    # author: rsshub puts the handle in <author> or <dc:creator>; fall back.
    author_handle = (
        (author_el.get_text(strip=True) if isinstance(author_el, Tag) else None)
        or fallback_handle
    )

    cashtags = tuple(sorted({m.group(1) for m in _CASHTAG_RE.finditer(content)}))
    mentions = tuple(sorted({m.group(1) for m in _MENTION_RE.finditer(content)}))

    return SocialPost(
        as_of=as_of,
        platform="x",
        post_id=post_id,
        author_handle=author_handle,
        author_name=channel_title,
        content=content,
        posted_at=posted_at,
        url=url,
        reply_to=reply_to,
        mentions=mentions,
        cashtags=cashtags,
        source="rsshub",
    )


# ── HTML stripping — rsshub embeds <p>, <br>, links in <description> ───────


_HTML_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Strip HTML tags. Cheap; we don't need DOM semantics."""
    # Replace common entities first so & doesn't leak through.
    s = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return _HTML_TAG_RE.sub("", s)
