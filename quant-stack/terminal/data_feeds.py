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
