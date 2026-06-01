"""Zacks rating client.

Public HTML at https://www.zacks.com/stock/quote/<SYMBOL>. No auth, no key.
Zacks updates ranks once a day pre-market, so polling every hour is plenty.

What we extract per ticker:

- Zacks rank (1-5) + textual label ("Strong Buy" through "Strong Sell")
- Price target consensus (when present)
- Style scores (V, G, M, VGM letter grades)
- Industry rank text

Zacks HTML is moderately stable but does shift. The parser is defensive
about missing elements — a missing field becomes ``None`` rather than
crashing. Any record that DOES parse is then pydantic-validated.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from types import TracebackType
from typing import Final

import httpx
from bs4 import BeautifulSoup

from ingestion._http import HTTPError, get_with_retry
from ingestion.schema import AnalystRating

_BASE_URL: Final[str] = "https://www.zacks.com"
_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (compatible; quant-stack/1.0; +https://github.com/)"
)
_DEFAULT_TIMEOUT_S: Final[float] = 10.0

# Map textual rank label → numeric (1=Strong Buy ... 5=Strong Sell).
_RANK_TEXT_TO_NUM: Final[dict[str, int]] = {
    "strong buy": 1,
    "buy": 2,
    "hold": 3,
    "sell": 4,
    "strong sell": 5,
}

_PRICE_TARGET_RE: Final[re.Pattern[str]] = re.compile(r"\$?([0-9,]+\.?[0-9]*)")
_INDUSTRY_RANK_RE: Final[re.Pattern[str]] = re.compile(r"(\d+)\s*/\s*\d+")


class ZacksError(RuntimeError):
    """Zacks request failed or returned an unparseable response."""


# ─────────────────────────────────────────────────────────────────────────────
#  Client
# ─────────────────────────────────────────────────────────────────────────────


class ZacksClient:
    """Sync httpx client for Zacks stock-quote pages."""

    def __init__(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self._client = httpx.Client(
            base_url=_BASE_URL,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
            timeout=timeout_s,
            follow_redirects=True,
        )

    def __enter__(self) -> ZacksClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._client.close()

    # ── public API ────────────────────────────────────────────────────────

    def get_rating(self, symbol: str) -> AnalystRating:
        """Pull the current Zacks rating snapshot for ``symbol``.

        Returns one :class:`AnalystRating` tagged with ``as_of=now(UTC)``.
        Persisted via ``ParquetStore.write_analyst_ratings`` which dedups
        to one snapshot per (symbol, day).
        """
        symbol = symbol.upper()
        html = self._get(f"/stock/quote/{symbol}")
        return _parse_rating(html, symbol=symbol, as_of=datetime.now(UTC))

    def _get(self, path: str) -> str:
        try:
            resp = get_with_retry(self._client, path)
        except HTTPError as e:
            msg = f"Zacks GET {path} failed: {e}"
            raise ZacksError(msg) from e
        return resp.text


# ─────────────────────────────────────────────────────────────────────────────
#  Parser — pure function
# ─────────────────────────────────────────────────────────────────────────────


def _parse_rating(html: str, *, symbol: str, as_of: datetime) -> AnalystRating:
    """Parse a Zacks stock-quote page into an AnalystRating.

    Most fields are best-effort — Zacks' selectors drift. Missing fields
    become None. The single required field is `source`.
    """
    soup = BeautifulSoup(html, "html.parser")

    rank_num: int | None = None
    rank_text: str | None = None

    # Numeric rank from .zr_rankbox .rank_view (e.g. "1")
    rank_el = soup.select_one(".zr_rankbox .rank_view")
    if rank_el:
        rank_str = rank_el.get_text(strip=True)
        match = re.search(r"[1-5]", rank_str)
        if match:
            rank_num = int(match.group(0))

    # Textual label from .zr_rankbox .rank_chip (e.g. "Strong Buy")
    chip_el = soup.select_one(".zr_rankbox .rank_chip")
    if chip_el:
        rank_text = chip_el.get_text(strip=True) or None
        if rank_num is None and rank_text:
            rank_num = _RANK_TEXT_TO_NUM.get(rank_text.lower())
    elif rank_num is not None:
        # Backfill text from number
        for label, num in _RANK_TEXT_TO_NUM.items():
            if num == rank_num:
                rank_text = label.title()
                break

    # Price target (consensus)
    price_target: float | None = None
    target_el = soup.select_one("#price_target_summary")
    if target_el:
        match = _PRICE_TARGET_RE.search(target_el.get_text(" ", strip=True))
        if match:
            try:
                price_target = float(match.group(1).replace(",", ""))
            except ValueError:
                price_target = None

    # Style scores — Value / Growth / Momentum / VGM
    style_value, style_growth, style_momentum, style_vgm = _parse_style_scores(soup)

    # Industry rank
    industry_rank, industry_rank_text = _parse_industry_rank(soup)

    return AnalystRating(
        as_of=as_of,
        symbol=symbol,
        rank=rank_num,
        rank_text=rank_text,
        price_target=price_target,
        style_score_value=style_value,
        style_score_growth=style_growth,
        style_score_momentum=style_momentum,
        style_score_vgm=style_vgm,
        industry_rank=industry_rank,
        industry_rank_text=industry_rank_text,
        source="zacks",
    )


# ── style-score helpers ────────────────────────────────────────────────────


_StyleScores = tuple[str | None, str | None, str | None, str | None]


def _parse_style_scores(soup: BeautifulSoup) -> _StyleScores:
    """Zacks style scores live in a .composite_val_rank or .scr_print_alert block.

    HTML varies; we try a couple of selectors and fall back to None.
    """
    value = growth = momentum = vgm = None

    # Modern layout: <div class="zr_rankbox style_score_box">...<span>A</span>...</div>
    for box in soup.select(".composite_val_rank, .scr_print_alert .alert_val"):
        text = box.get_text(" ", strip=True)
        if not text:
            continue
        # Each box has labels like "Value: A" or just letter grades A-F
        if "value" in text.lower():
            value = _first_letter_grade(text)
        elif "growth" in text.lower():
            growth = _first_letter_grade(text)
        elif "momentum" in text.lower():
            momentum = _first_letter_grade(text)
        elif "vgm" in text.lower():
            vgm = _first_letter_grade(text)

    return value, growth, momentum, vgm


def _first_letter_grade(text: str) -> str | None:
    """Return the first standalone A-F letter found in ``text``."""
    match = re.search(r"\b([A-F])\b", text)
    return match.group(1) if match else None


def _parse_industry_rank(soup: BeautifulSoup) -> tuple[int | None, str | None]:
    """Industry rank shows as 'XX / YYY' (e.g. '25 / 250').

    Returns (numerator_int, full_text). Either may be None.
    """
    for el in soup.select(".sector_industry_rank, .industry_rank, .sector_rank"):
        text = el.get_text(" ", strip=True)
        match = _INDUSTRY_RANK_RE.search(text)
        if match:
            try:
                return int(match.group(1)), text
            except ValueError:
                return None, text
    return None, None
