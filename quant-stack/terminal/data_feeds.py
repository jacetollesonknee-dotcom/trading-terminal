"""
data_feeds.py — Real-time financial data aggregator.

Sources: Yahoo Finance, Zacks, OpenInsider, Kalshi, and 8 news outlets.
Adds short-TTL caching, retries, and timeouts on every external call.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable

import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_HTTP = httpx.Client(
    headers=HEADERS,
    timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
    follow_redirects=True,
    http2=False,
)

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="feeds")


# ── tiny TTL cache ───────────────────────────────────────────────────────
class _TTLCache:
    def __init__(self):
        self._store: dict = {}
        self._lock = threading.Lock()

    def get(self, key: str, ttl: float):
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
        if entry and now - entry[0] < ttl:
            return entry[1]
        return None

    def set(self, key: str, value):
        with self._lock:
            self._store[key] = (time.time(), value)


_cache = _TTLCache()


def _cached(key: str, ttl: float, fn: Callable):
    hit = _cache.get(key, ttl)
    if hit is not None:
        return hit
    val = fn()
    _cache.set(key, val)
    return val


def _retry_get(url: str, *, retries: int = 2, backoff: float = 0.6, **kw) -> httpx.Response | None:
    """GET with retry on transient errors. Returns None on hard failure."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = _HTTP.get(url, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                last_exc = httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
            else:
                return r
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
        time.sleep(backoff * (2 ** attempt))
    return None


# ═══════════════════════════════════════════════════════════════════════
#  YAHOO FINANCE
# ═══════════════════════════════════════════════════════════════════════

def yahoo_quote(symbol: str) -> dict:
    sym = symbol.upper()
    cached = _cache.get(f"q:{sym}", ttl=10)
    if cached is not None:
        return cached
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        r = _retry_get(url, params={"interval": "1d", "range": "1d", "includePrePost": "true"})
        if r is None:
            return {"error": "network", "symbol": sym, "source": "yahoo"}
        data = r.json()
        result = data["chart"]["result"][0]
        meta = result["meta"]
        price = meta.get("regularMarketPrice", 0) or 0
        prev = meta.get("chartPreviousClose", price) or price
        change = round(price - prev, 2)
        change_pct = round((change / prev) * 100, 2) if prev else 0
        out = {
            "symbol": sym,
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "volume": meta.get("regularMarketVolume", 0) or 0,
            "high": meta.get("regularMarketDayHigh", price) or price,
            "low": meta.get("regularMarketDayLow", price) or price,
            "open": meta.get("regularMarketOpen", price) or price,
            "close": meta.get("previousClose", prev) or prev,
            "fifty_two_week_high": meta.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": meta.get("fiftyTwoWeekLow"),
            "market_cap": meta.get("marketCap"),
            "exchange": meta.get("exchangeName", ""),
            "currency": meta.get("currency", "USD"),
            "instrument_type": meta.get("instrumentType", ""),
            "source": "yahoo",
            "live": True,
            "as_of": datetime.now().isoformat(),
        }
        _cache.set(f"q:{sym}", out)
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "symbol": sym, "source": "yahoo"}


def yahoo_history(symbol: str, period: str = "3mo", interval: str = "1d") -> list:
    sym = symbol.upper()
    key = f"h:{sym}:{period}:{interval}"
    cached = _cache.get(key, ttl=60)
    if cached is not None:
        return cached
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        r = _retry_get(url, params={"interval": interval, "range": period})
        if r is None:
            return [{"error": "network"}]
        data = r.json()
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        ohlcv = result.get("indicators", {}).get("quote", [{}])[0]
        history = []
        for i, ts in enumerate(timestamps):
            close_v = ohlcv.get("close", [None] * len(timestamps))[i]
            if close_v is None:
                continue
            history.append({
                "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                "open": round(ohlcv.get("open", [0])[i] or 0, 2),
                "high": round(ohlcv.get("high", [0])[i] or 0, 2),
                "low": round(ohlcv.get("low", [0])[i] or 0, 2),
                "close": round(close_v, 2),
                "volume": ohlcv.get("volume", [0])[i] or 0,
            })
        _cache.set(key, history)
        return history
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]


def yahoo_fundamentals(symbol: str) -> dict:
    sym = symbol.upper()
    cached = _cache.get(f"f:{sym}", ttl=600)
    if cached is not None:
        return cached
    try:
        url = f"https://finance.yahoo.com/quote/{sym}/"
        r = _retry_get(url)
        if r is None:
            return {"error": "network", "symbol": sym}
        soup = BeautifulSoup(r.text, "html.parser")
        stats = {}
        for row in soup.select("tr"):
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                if key and val and len(key) < 60:
                    stats[key] = val
        out = {"symbol": sym, "fundamentals": stats, "source": "yahoo"}
        _cache.set(f"f:{sym}", out)
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "symbol": sym}


def yahoo_trending() -> list:
    cached = _cache.get("trend", ttl=300)
    if cached is not None:
        return cached
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        r = _retry_get(url)
        if r is None:
            return []
        data = r.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        out = [q["symbol"] for q in quotes[:25] if "symbol" in q]
        _cache.set("trend", out)
        return out
    except Exception:  # noqa: BLE001
        return []


def yahoo_movers() -> dict:
    cached = _cache.get("movers", ttl=120)
    if cached is not None:
        return cached
    try:
        results = {"gainers": [], "losers": [], "active": []}
        for category in ["gainers", "losers", "most_active"]:
            url = (
                "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
                f"?scrIds=day_{category}&count=10"
            )
            r = _retry_get(url)
            if r is None:
                continue
            data = r.json()
            quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
            key = "active" if category == "most_active" else category
            for q in quotes[:10]:
                results[key].append({
                    "symbol": q.get("symbol", ""),
                    "name": q.get("shortName", ""),
                    "price": q.get("regularMarketPrice", 0),
                    "change_pct": round(q.get("regularMarketChangePercent", 0) or 0, 2),
                    "volume": q.get("regularMarketVolume", 0),
                })
        _cache.set("movers", results)
        return results
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def yahoo_search(query: str) -> list:
    """Symbol/quote autocomplete."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        r = _retry_get(url, params={"q": query, "quotesCount": 8, "newsCount": 0})
        if r is None:
            return []
        items = r.json().get("quotes", [])
        return [{
            "symbol": q.get("symbol"),
            "name": q.get("shortname") or q.get("longname"),
            "exchange": q.get("exchange"),
            "type": q.get("quoteType"),
        } for q in items if q.get("symbol")]
    except Exception:  # noqa: BLE001
        return []


# ═══════════════════════════════════════════════════════════════════════
#  ZACKS
# ═══════════════════════════════════════════════════════════════════════

def zacks_rating(symbol: str) -> dict:
    sym = symbol.upper()
    cached = _cache.get(f"z:{sym}", ttl=900)
    if cached is not None:
        return cached
    try:
        url = f"https://www.zacks.com/stock/quote/{sym}"
        r = _retry_get(url, headers={**HEADERS, "Accept": "text/html"})
        if r is None:
            return {"error": "network", "symbol": sym, "source": "zacks"}
        soup = BeautifulSoup(r.text, "html.parser")
        rating = "N/A"
        rating_text = "N/A"
        rank_el = soup.select_one(".zr_rankbox .rank_view")
        if rank_el:
            rating = rank_el.get_text(strip=True)
        rank_desc = soup.select_one(".zr_rankbox .rank_chip")
        if rank_desc:
            rating_text = rank_desc.get_text(strip=True)
        stats = {}
        for st in soup.select(".key_stat_title"):
            key = st.get_text(strip=True)
            val_el = st.find_next_sibling()
            if val_el:
                stats[key] = val_el.get_text(strip=True)
        target = None
        target_el = soup.select_one("#price_target_summary")
        if target_el:
            target = target_el.get_text(strip=True)
        out = {
            "symbol": sym,
            "zacks_rank": rating,
            "rank_text": rating_text,
            "stats": stats,
            "price_target": target,
            "source": "zacks",
            "url": url,
        }
        _cache.set(f"z:{sym}", out)
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "symbol": sym, "source": "zacks"}


# ═══════════════════════════════════════════════════════════════════════
#  OPENINSIDER
# ═══════════════════════════════════════════════════════════════════════

def _parse_insider_table(soup) -> list:
    table = soup.select_one("table.tinytable")
    if not table:
        return []
    trades = []
    for row in table.select("tbody tr")[:30]:
        cells = row.find_all("td")
        if len(cells) < 12:
            continue
        try:
            trades.append({
                "filing_date": cells[1].get_text(strip=True),
                "trade_date": cells[2].get_text(strip=True),
                "ticker": cells[3].get_text(strip=True),
                "company": cells[4].get_text(strip=True),
                "insider_name": cells[5].get_text(strip=True),
                "title": cells[6].get_text(strip=True),
                "trade_type": cells[7].get_text(strip=True),
                "price": cells[8].get_text(strip=True),
                "qty": cells[9].get_text(strip=True),
                "owned": cells[10].get_text(strip=True),
                "value": cells[11].get_text(strip=True),
                "source": "openinsider",
            })
        except (IndexError, AttributeError):
            continue
    return trades


def openinsider_latest(symbol: str | None = None) -> list:
    cache_key = f"oi:{symbol or 'latest'}"
    cached = _cache.get(cache_key, ttl=120)
    if cached is not None:
        return cached
    try:
        if symbol:
            url = (
                f"http://openinsider.com/screener?s={symbol.upper()}"
                "&o=&pl=&ph=&ll=&lh=&fd=30&fdr=&td=0&tdr=&feession="
                "&at=&a=&c=&cnt=25&page=1"
            )
        else:
            url = "http://openinsider.com/latest-insider-trading"
        r = _retry_get(url)
        if r is None:
            return [{"error": "network", "source": "openinsider"}]
        soup = BeautifulSoup(r.text, "html.parser")
        trades = _parse_insider_table(soup)
        _cache.set(cache_key, trades)
        return trades
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e), "source": "openinsider"}]


def openinsider_top_buys() -> list:
    cached = _cache.get("oi:top", ttl=600)
    if cached is not None:
        return cached
    try:
        url = "http://openinsider.com/top-insider-purchases-of-the-month"
        r = _retry_get(url)
        if r is None:
            return [{"error": "network"}]
        soup = BeautifulSoup(r.text, "html.parser")
        trades = _parse_insider_table(soup)
        _cache.set("oi:top", trades)
        return trades
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]


# ═══════════════════════════════════════════════════════════════════════
#  KALSHI prediction markets
# ═══════════════════════════════════════════════════════════════════════

def kalshi_markets(query: str | None = None, category: str = "economics") -> list:
    cache_key = f"k:{query or 'all'}:{category}"
    cached = _cache.get(cache_key, ttl=120)
    if cached is not None:
        return cached
    try:
        url = "https://api.elections.kalshi.com/trade-api/v2/markets"
        params = {"limit": 25, "status": "open"}
        if query:
            params["ticker"] = query.upper()
        r = _retry_get(url, params=params,
                       headers={**HEADERS, "Accept": "application/json"})
        if r is None:
            return [{"error": "network", "source": "kalshi"}]
        markets = r.json().get("markets", [])
        results = []
        for m in markets[:25]:
            results.append({
                "ticker": m.get("ticker", ""),
                "title": m.get("title", ""),
                "subtitle": m.get("subtitle", ""),
                "yes_price": (m.get("yes_bid", 0) / 100) if m.get("yes_bid") else None,
                "no_price": (m.get("no_bid", 0) / 100) if m.get("no_bid") else None,
                "volume": m.get("volume", 0),
                "open_interest": m.get("open_interest", 0),
                "close_time": m.get("close_time", ""),
                "category": m.get("category", ""),
                "status": m.get("status", ""),
                "source": "kalshi",
            })
        _cache.set(cache_key, results)
        return results
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e), "source": "kalshi"}]


def kalshi_events(category: str | None = None) -> list:
    try:
        url = "https://api.elections.kalshi.com/trade-api/v2/events"
        params = {"limit": 25, "status": "open"}
        if category:
            params["series_ticker"] = category.upper()
        r = _retry_get(url, params=params,
                       headers={**HEADERS, "Accept": "application/json"})
        if r is None:
            return [{"error": "network", "source": "kalshi"}]
        events = r.json().get("events", [])
        return [{
            "ticker": e.get("event_ticker", ""),
            "title": e.get("title", ""),
            "category": e.get("category", ""),
            "markets_count": len(e.get("markets", [])),
            "source": "kalshi",
        } for e in events[:25]]
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e), "source": "kalshi"}]


# ═══════════════════════════════════════════════════════════════════════
#  NEWS aggregator
# ═══════════════════════════════════════════════════════════════════════

def _clean_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:300]


def _parse_rss(url: str, source: str) -> list:
    try:
        r = _retry_get(url)
        if r is None:
            return []
        soup = BeautifulSoup(r.text, "xml")
        items = soup.find_all("item")
        articles = []
        for item in items[:18]:
            title = item.find("title")
            link = item.find("link")
            pub = item.find("pubDate")
            desc = item.find("description")
            articles.append({
                "title": title.get_text(strip=True) if title else "",
                "url": link.get_text(strip=True) if link else "",
                "published": pub.get_text(strip=True) if pub else "",
                "summary": _clean_html(desc.get_text() if desc else ""),
                "source": source,
            })
        return articles
    except Exception:  # noqa: BLE001
        return []


_NEWS_SOURCES: list[tuple[str, str]] = [
    ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "CNBC"),
    ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",  "CNBC Markets"),
    ("https://feeds.marketwatch.com/marketwatch/topstories/",                                 "MarketWatch"),
    ("https://feeds.marketwatch.com/marketwatch/marketpulse/",                                "MarketWatch Pulse"),
    ("https://feeds.a.dj.com/rss/RSSMarketsMain.xml",                                         "WSJ"),
    ("https://www.benzinga.com/feed",                                                         "Benzinga"),
    ("https://finance.yahoo.com/news/rssindex",                                               "Yahoo Finance"),
    ("https://seekingalpha.com/market_currents.xml",                                          "Seeking Alpha"),
    ("https://www.investing.com/rss/news.rss",                                                "Investing.com"),
]


def news_for_symbol(symbol: str) -> list:
    cached = _cache.get(f"n:{symbol.upper()}", ttl=180)
    if cached is not None:
        return cached
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    out = _parse_rss(url, f"Yahoo ({symbol.upper()})")
    _cache.set(f"n:{symbol.upper()}", out)
    return out


def aggregate_news(limit: int = 50) -> list:
    cache_key = f"news_agg:{limit}"
    cached = _cache.get(cache_key, ttl=120)
    if cached is not None:
        return cached
    futures = {_executor.submit(_parse_rss, url, src): src for url, src in _NEWS_SOURCES}
    all_news: list = []
    for fut in as_completed(futures, timeout=15):
        try:
            all_news.extend(fut.result() or [])
        except Exception:  # noqa: BLE001
            continue
    # Dedup: hash on first 60 chars of normalized title
    seen = set()
    unique = []
    for a in all_news:
        title = (a.get("title") or "").lower().strip()
        key = re.sub(r"[^a-z0-9]+", "", title)[:60]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(a)
    out = unique[:limit]
    _cache.set(cache_key, out)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════

_NAME_MAP = {
    "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "Nasdaq",
    "^RUT": "Russell 2000", "^VIX": "VIX",
    "ES=F": "S&P Futures", "NQ=F": "Nasdaq Futures",
    "YM=F": "Dow Futures", "CL=F": "Crude Oil", "GC=F": "Gold",
    "SI=F": "Silver", "NG=F": "Nat Gas",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "DX-Y.NYB": "DXY", "^TNX": "10Y Yield",
}


def market_overview() -> dict:
    cached = _cache.get("mkt", ttl=30)
    if cached is not None:
        return cached
    symbols = {
        "indices":   ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^TNX"],
        "futures":   ["ES=F", "NQ=F", "YM=F", "CL=F", "GC=F", "SI=F"],
        "crypto":    ["BTC-USD", "ETH-USD", "SOL-USD"],
    }
    result: dict = {}
    futures_map: dict = {}
    for category, syms in symbols.items():
        result[category] = []
        for sym in syms:
            futures_map[_executor.submit(yahoo_quote, sym)] = (category, sym)

    for fut in as_completed(futures_map, timeout=15):
        category, sym = futures_map[fut]
        try:
            q = fut.result()
            if "error" in q:
                continue
            q["name"] = _NAME_MAP.get(sym, sym)
            result[category].append(q)
        except Exception:  # noqa: BLE001
            continue

    # Preserve original ordering
    for category, syms in symbols.items():
        ordering = {s: i for i, s in enumerate(syms)}
        result[category].sort(key=lambda q: ordering.get(q.get("symbol", ""), 999))

    _cache.set("mkt", result)
    return result


# ═══════════════════════════════════════════════════════════════════════
#  FEAR & GREED
# ═══════════════════════════════════════════════════════════════════════

def fear_greed_index() -> dict:
    cached = _cache.get("fg", ttl=600)
    if cached is not None:
        return cached
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = _retry_get(url, headers={**HEADERS, "Accept": "application/json"})
        if r is None:
            return {"error": "network", "source": "CNN Fear & Greed"}
        fg = r.json().get("fear_and_greed", {})
        out = {
            "score": round(fg.get("score", 0) or 0, 1),
            "rating": fg.get("rating", "N/A"),
            "previous_close": round(fg.get("previous_close", 0) or 0, 1),
            "week_ago": round(fg.get("previous_1_week", 0) or 0, 1),
            "month_ago": round(fg.get("previous_1_month", 0) or 0, 1),
            "year_ago": round(fg.get("previous_1_year", 0) or 0, 1),
            "source": "CNN Fear & Greed",
        }
        _cache.set("fg", out)
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "source": "CNN Fear & Greed"}


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: full research bundle
# ═══════════════════════════════════════════════════════════════════════

def full_research(symbol: str) -> dict:
    """Bundle quote + history + fundamentals + zacks + insider + news in parallel."""
    sym = symbol.upper()
    tasks = [
        ("quote",          yahoo_quote,          (sym,)),
        ("history",        yahoo_history,        (sym, "3mo", "1d")),
        ("fundamentals",   yahoo_fundamentals,   (sym,)),
        ("zacks",          zacks_rating,         (sym,)),
        ("insider_trades", openinsider_latest,   (sym,)),
        ("news",           news_for_symbol,      (sym,)),
    ]
    futures_map = {_executor.submit(fn, *args): key for key, fn, args in tasks}
    results: dict = {"symbol": sym}
    for fut in as_completed(futures_map, timeout=20):
        key = futures_map[fut]
        try:
            results[key] = fut.result()
        except Exception as e:  # noqa: BLE001
            results[key] = {"error": str(e)}
    return results


# ═══════════════════════════════════════════════════════════════════════
#  MACRO RADAR — global cross-asset indicators that signal market direction
# ═══════════════════════════════════════════════════════════════════════
#
# Each indicator carries a static "read" describing how it is normally
# interpreted, plus a `bias` function that turns the live quote into a
# risk-on / risk-off / neutral tilt. This is what powers the dashboard's
# per-indicator explanations and the morning macro sentiment brief.

def _pct(q: dict) -> float:
    return q.get("change_pct") or 0.0


def _price(q: dict) -> float:
    return q.get("price") or 0.0


# Group → list of indicator specs.
# invert=True means "a rising value is risk-OFF" (e.g. VIX, gold, bonds price).
MACRO_GROUPS: dict = {
    "Volatility & Fear": [
        {"symbol": "^VIX", "name": "VIX", "unit": "",
         "read": "The market's 30-day fear gauge (implied vol on S&P 500 options). "
                 "Under ~15 = complacent/risk-on; 20-30 = nervous; above 30 = fear/de-risking. "
                 "Spikes usually coincide with equity sell-offs.",
         "invert": True, "hot": lambda q: _price(q) >= 25},
        {"symbol": "^VVIX", "name": "VVIX (vol-of-vol)", "unit": "",
         "read": "Volatility OF the VIX — how frantically traders are bidding for VIX options. "
                 "Rising VVIX with a calm VIX warns that a volatility spike is being hedged for. "
                 "Above ~110 signals stress building under the surface.",
         "invert": True, "hot": lambda q: _price(q) >= 110},
        {"symbol": "^VXN", "name": "VXN (Nasdaq vol)", "unit": "",
         "read": "Implied volatility on the Nasdaq-100. Tech-heavy fear gauge; leads the VIX "
                 "when growth/mega-cap names are under pressure.",
         "invert": True, "hot": lambda q: _price(q) >= 28},
    ],
    "Rates & Bonds": [
        {"symbol": "^TNX", "name": "US 10Y Treasury Yield", "unit": "%",
         "read": "The world's benchmark discount rate. Rising yields pressure long-duration/growth "
                 "stocks and gold; falling yields ease financial conditions. Watch the SPEED of the "
                 "move more than the level — fast spikes break risk assets.",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 3},
        {"symbol": "^TYX", "name": "US 30Y Treasury Yield", "unit": "%",
         "read": "The long bond. Reflects long-run growth + inflation + term-premium expectations. "
                 "A steepening 30Y vs 10Y often flags inflation or fiscal worry.",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 3},
        {"symbol": "^FVX", "name": "US 5Y Treasury Yield", "unit": "%",
         "read": "The belly of the curve — most sensitive to the Fed's expected policy path. "
                 "Leads repricing of rate-cut/hike odds.",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 3},
        {"symbol": "^IRX", "name": "13-Week T-Bill Yield", "unit": "%",
         "read": "The front end ≈ where the market thinks the Fed funds rate is going near-term. "
                 "Anchors cash yields; inversion vs the 10Y (front > long) is a recession flag.",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 4},
        {"symbol": "TLT", "name": "20Y+ Treasury ETF", "unit": "$",
         "read": "Price proxy for long bonds (moves OPPOSITE to yields). Rising TLT = flight to "
                 "safety / lower rates = usually risk-off for the economy but a tailwind for duration.",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 1.5},
        {"symbol": "HYG", "name": "High-Yield Credit ETF", "unit": "$",
         "read": "Junk-bond ETF — the canary for credit stress. When HYG rolls over while stocks "
                 "hold up, it warns that the credit market sees trouble equities haven't priced yet.",
         "invert": False, "hot": lambda q: _pct(q) <= -1},
    ],
    "FX & Dollar": [
        {"symbol": "DX-Y.NYB", "name": "US Dollar Index (DXY)", "unit": "",
         "read": "The dollar vs a basket of majors. A strong dollar tightens global financial "
                 "conditions, pressures commodities, EM, and US multinationals' earnings. "
                 "Falling DXY is broadly risk-on.",
         "invert": True, "hot": lambda q: abs(_pct(q)) >= 0.6},
        {"symbol": "JPY=X", "name": "USD/JPY (Yen)", "unit": "",
         "read": "The world's carry-trade funding pair. A fast rise (weak yen) can fuel risk appetite, "
                 "but a sudden DROP (yen strengthening) often means a carry-trade unwind — a classic "
                 "trigger for global de-risking (see Aug 2024).",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 0.8},
        {"symbol": "EURUSD=X", "name": "EUR/USD", "unit": "",
         "read": "The anti-dollar. Rising EUR/USD usually confirms a weakening dollar and easier "
                 "global liquidity (risk-on).",
         "invert": False, "hot": lambda q: abs(_pct(q)) >= 0.6},
        {"symbol": "KRW=X", "name": "USD/KRW (Won)", "unit": "",
         "read": "The Korean won is a high-beta EM/Asia risk barometer given Korea's export & "
                 "semiconductor exposure. A weakening won (rising USD/KRW) signals Asia risk-off.",
         "invert": True, "hot": lambda q: abs(_pct(q)) >= 0.7},
    ],
    "Commodities": [
        {"symbol": "GC=F", "name": "Gold", "unit": "$",
         "read": "The premier safe-haven & real-rates / debasement hedge. Rising gold WITH rising "
                 "yields is unusual and flags fear or de-dollarization; rising gold with falling "
                 "yields is the classic risk-off flight to safety.",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 1.5},
        {"symbol": "CL=F", "name": "WTI Crude Oil", "unit": "$",
         "read": "US benchmark oil. A demand & growth proxy — but sharp spikes are an inflation/"
                 "stagflation risk that squeezes consumers and can force central banks tighter.",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 3},
        {"symbol": "BZ=F", "name": "Brent Crude Oil", "unit": "$",
         "read": "The global oil benchmark. Brent minus WTI (the spread) reflects geopolitical & "
                 "shipping risk; a widening spread often means a supply scare abroad.",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 3},
        {"symbol": "HG=F", "name": "Copper (Dr. Copper)", "unit": "$",
         "read": "'Dr. Copper' — used everywhere in industry, so it reads the pulse of global growth. "
                 "Rising copper = expansion/reflation; falling copper warns of a slowdown.",
         "invert": False, "hot": lambda q: abs(_pct(q)) >= 2},
        {"symbol": "SI=F", "name": "Silver", "unit": "$",
         "read": "Half precious-metal haven, half industrial. Outperforming gold in a rally signals "
                 "reflation/risk appetite; lagging gold signals defensive haven demand.",
         "invert": None, "hot": lambda q: abs(_pct(q)) >= 2},
    ],
    "Global Equity": [
        {"symbol": "^GSPC", "name": "S&P 500", "unit": "",
         "read": "The US large-cap benchmark — the market everyone else is measured against.",
         "invert": False, "hot": lambda q: abs(_pct(q)) >= 1},
        {"symbol": "^IXIC", "name": "Nasdaq Composite", "unit": "",
         "read": "Growth / tech / long-duration equity. Leads on the way up AND down; "
                 "underperformance vs the S&P signals a defensive rotation out of risk.",
         "invert": False, "hot": lambda q: abs(_pct(q)) >= 1.2},
        {"symbol": "^RUT", "name": "Russell 2000 (small caps)", "unit": "",
         "read": "Domestic small caps — most sensitive to the US economy, credit, and rates. "
                 "Small-cap leadership = healthy risk-on breadth; lagging = narrow, fragile rally.",
         "invert": False, "hot": lambda q: abs(_pct(q)) >= 1.2},
        {"symbol": "^N225", "name": "Nikkei 225 (Japan)", "unit": "",
         "read": "Japan's benchmark — tightly linked to USD/JPY. Because Tokyo trades before the US, "
                 "it's an overnight tell for global risk sentiment.",
         "invert": False, "hot": lambda q: abs(_pct(q)) >= 1.5},
        {"symbol": "^KS11", "name": "KOSPI (South Korea)", "unit": "",
         "read": "Korea's index is heavy in semiconductors & exports — a leading indicator for the "
                 "global tech cycle and Asian trade. Weakness here often precedes tech weakness in the US.",
         "invert": False, "hot": lambda q: abs(_pct(q)) >= 1.5},
        {"symbol": "^HSI", "name": "Hang Seng (Hong Kong)", "unit": "",
         "read": "The main window into China risk sentiment and global EM appetite.",
         "invert": False, "hot": lambda q: abs(_pct(q)) >= 1.5},
    ],
}

# Flatten for quick symbol → spec lookup (used by explain-indicator).
MACRO_SPECS: dict = {
    spec["symbol"]: {**spec, "group": group}
    for group, specs in MACRO_GROUPS.items()
    for spec in specs
}


def _indicator_bias(spec: dict, q: dict) -> str:
    """Turn a live quote into a coarse risk tilt for the dashboard."""
    pct = _pct(q)
    if abs(pct) < 0.15:
        return "neutral"
    invert = spec.get("invert")
    if invert is None:
        return "neutral"          # direction is context-dependent (rates, gold, yen)
    up_is_risk_on = not invert
    rising = pct > 0
    if rising == up_is_risk_on:
        return "risk-on"
    return "risk-off"


def macro_dashboard() -> dict:
    """Fetch every macro-radar indicator in parallel, grouped, with metadata."""
    cached = _cache.get("macro", ttl=30)
    if cached is not None:
        return cached

    all_symbols = list(MACRO_SPECS.keys())
    futures = {_executor.submit(yahoo_quote, s): s for s in all_symbols}
    quotes: dict = {}
    for fut in as_completed(futures, timeout=18):
        sym = futures[fut]
        try:
            q = fut.result()
            if isinstance(q, dict) and "error" not in q:
                quotes[sym] = q
        except Exception:  # noqa: BLE001
            continue

    groups_out: list = []
    tally = {"risk-on": 0, "risk-off": 0, "neutral": 0}
    for group, specs in MACRO_GROUPS.items():
        items = []
        for spec in specs:
            q = quotes.get(spec["symbol"])
            if not q:
                continue
            bias = _indicator_bias(spec, q)
            tally[bias] = tally.get(bias, 0) + 1
            hot = False
            try:
                hot = bool(spec.get("hot") and spec["hot"](q))
            except Exception:  # noqa: BLE001
                hot = False
            items.append({
                "symbol": spec["symbol"],
                "name": spec["name"],
                "unit": spec.get("unit", ""),
                "price": q.get("price"),
                "change": q.get("change"),
                "change_pct": q.get("change_pct"),
                "read": spec["read"],
                "bias": bias,
                "hot": hot,
            })
        if items:
            groups_out.append({"group": group, "indicators": items})

    scored = tally["risk-on"] + tally["risk-off"]
    if scored == 0:
        tilt, tilt_score = "neutral", 0.0
    else:
        tilt_score = round((tally["risk-on"] - tally["risk-off"]) / scored, 2)
        if tilt_score >= 0.25:
            tilt = "risk-on"
        elif tilt_score <= -0.25:
            tilt = "risk-off"
        else:
            tilt = "mixed"

    out = {
        "groups": groups_out,
        "tilt": tilt,
        "tilt_score": tilt_score,
        "tally": tally,
        "as_of": datetime.now().isoformat(),
    }
    _cache.set("macro", out)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  SECTOR ROTATION — which sectors money is flowing into (by move + volume)
# ═══════════════════════════════════════════════════════════════════════

SECTOR_ETFS: list = [
    {"symbol": "XLK", "name": "Technology", "risk": "cyclical"},
    {"symbol": "XLC", "name": "Communication Svcs", "risk": "cyclical"},
    {"symbol": "XLY", "name": "Consumer Discretionary", "risk": "cyclical"},
    {"symbol": "XLF", "name": "Financials", "risk": "cyclical"},
    {"symbol": "XLI", "name": "Industrials", "risk": "cyclical"},
    {"symbol": "XLB", "name": "Materials", "risk": "cyclical"},
    {"symbol": "XLE", "name": "Energy", "risk": "cyclical"},
    {"symbol": "XLV", "name": "Health Care", "risk": "defensive"},
    {"symbol": "XLP", "name": "Consumer Staples", "risk": "defensive"},
    {"symbol": "XLU", "name": "Utilities", "risk": "defensive"},
    {"symbol": "XLRE", "name": "Real Estate", "risk": "defensive"},
]


def sector_rotation() -> dict:
    """Rank the 11 SPDR sectors by performance, with volume vs its own average.

    Volume ratio (today's volume / recent average) shows CONVICTION — a sector
    up on heavy volume is real rotation IN; up on light volume is a drift.
    """
    cached = _cache.get("sectors", ttl=60)
    if cached is not None:
        return cached

    def _one(spec: dict) -> dict | None:
        q = yahoo_quote(spec["symbol"])
        if not isinstance(q, dict) or "error" in q:
            return None
        hist = yahoo_history(spec["symbol"], period="1mo", interval="1d")
        vols = [h.get("volume", 0) for h in (hist or []) if isinstance(h, dict) and h.get("volume")]
        avg_vol = sum(vols[-20:]) / len(vols[-20:]) if vols else 0
        today_vol = q.get("volume", 0) or 0
        vol_ratio = round(today_vol / avg_vol, 2) if avg_vol else None
        return {
            "symbol": spec["symbol"],
            "name": spec["name"],
            "risk": spec["risk"],
            "price": q.get("price"),
            "change_pct": q.get("change_pct") or 0,
            "volume": today_vol,
            "avg_volume": int(avg_vol),
            "vol_ratio": vol_ratio,
        }

    futures = {_executor.submit(_one, spec): spec["symbol"] for spec in SECTOR_ETFS}
    sectors: list = []
    for fut in as_completed(futures, timeout=20):
        try:
            r = fut.result()
            if r:
                sectors.append(r)
        except Exception:  # noqa: BLE001
            continue

    sectors.sort(key=lambda s: s.get("change_pct", 0), reverse=True)

    def _avg(items):
        vals = [s["change_pct"] for s in items]
        return round(sum(vals) / len(vals), 2) if vals else 0

    cyc = [s for s in sectors if s["risk"] == "cyclical"]
    dfn = [s for s in sectors if s["risk"] == "defensive"]
    breadth = _avg(cyc) - _avg(dfn)
    if breadth >= 0.3:
        leadership = "cyclical"   # offense leading = risk-on rotation
    elif breadth <= -0.3:
        leadership = "defensive"  # defense leading = risk-off rotation
    else:
        leadership = "mixed"

    out = {
        "sectors": sectors,
        "leadership": leadership,
        "cyclical_avg": _avg(cyc),
        "defensive_avg": _avg(dfn),
        "as_of": datetime.now().isoformat(),
    }
    _cache.set("sectors", out)
    return out
