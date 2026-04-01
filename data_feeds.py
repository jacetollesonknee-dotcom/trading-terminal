"""
data_feeds.py – Real-time financial data aggregator
Pulls from Yahoo Finance, Zacks, OpenInsider, Kalshi, and multiple news outlets.
"""

import os, json, time, re, traceback
from datetime import datetime, timedelta
from functools import lru_cache
import threading
import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# ═══════════════════════════════════════════════════════════════════════
#  YAHOO FINANCE  –  quotes, fundamentals, history, movers
# ═══════════════════════════════════════════════════════════════════════

def yahoo_quote(symbol: str) -> dict:
    """Get real-time quote from Yahoo Finance."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": "1d", "range": "1d", "includePrePost": "true"}
        r = httpx.get(url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
        result = data["chart"]["result"][0]
        meta = result["meta"]
        price = meta.get("regularMarketPrice", 0)
        prev = meta.get("chartPreviousClose", price)
        change = round(price - prev, 2)
        change_pct = round((change / prev) * 100, 2) if prev else 0
        return {
            "symbol": symbol.upper(),
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "volume": meta.get("regularMarketVolume", 0),
            "high": meta.get("regularMarketDayHigh", price),
            "low": meta.get("regularMarketDayLow", price),
            "open": meta.get("regularMarketOpen", price),
            "close": meta.get("previousClose", prev),
            "market_cap": meta.get("marketCap", None),
            "exchange": meta.get("exchangeName", ""),
            "currency": meta.get("currency", "USD"),
            "source": "yahoo",
            "live": True
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol, "source": "yahoo"}


def yahoo_history(symbol: str, period: str = "3mo", interval: str = "1d") -> list:
    """Get historical OHLCV data from Yahoo Finance."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": interval, "range": period}
        r = httpx.get(url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        ohlcv = result["indicators"]["quote"][0]
        history = []
        for i, ts in enumerate(timestamps):
            if ohlcv["close"][i] is None:
                continue
            history.append({
                "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                "open": round(ohlcv["open"][i] or 0, 2),
                "high": round(ohlcv["high"][i] or 0, 2),
                "low": round(ohlcv["low"][i] or 0, 2),
                "close": round(ohlcv["close"][i] or 0, 2),
                "volume": ohlcv["volume"][i] or 0
            })
        return history
    except Exception as e:
        return [{"error": str(e)}]


def yahoo_fundamentals(symbol: str) -> dict:
    """Scrape key fundamentals from Yahoo Finance summary page."""
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/"
        r = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(r.text, "html.parser")

        stats = {}
        # Try to extract from the data tables
        for row in soup.select("tr"):
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                stats[key] = val

        return {"symbol": symbol.upper(), "fundamentals": stats, "source": "yahoo"}
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


def yahoo_trending() -> list:
    """Get trending tickers from Yahoo Finance."""
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        r = httpx.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        return [q["symbol"] for q in quotes[:20]]
    except Exception:
        return []


def yahoo_movers() -> dict:
    """Get market movers – gainers, losers, most active."""
    try:
        results = {"gainers": [], "losers": [], "active": []}
        for category in ["gainers", "losers", "most_active"]:
            url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_{category}&count=10"
            r = httpx.get(url, headers=HEADERS, timeout=10)
            data = r.json()
            quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
            key = "active" if category == "most_active" else category
            for q in quotes[:10]:
                results[key].append({
                    "symbol": q.get("symbol", ""),
                    "name": q.get("shortName", ""),
                    "price": q.get("regularMarketPrice", 0),
                    "change_pct": round(q.get("regularMarketChangePercent", 0), 2),
                    "volume": q.get("regularMarketVolume", 0)
                })
        return results
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  ZACKS  –  ratings and research
# ═══════════════════════════════════════════════════════════════════════

def zacks_rating(symbol: str) -> dict:
    """Scrape Zacks rank and data for a symbol."""
    try:
        url = f"https://www.zacks.com/stock/quote/{symbol}"
        r = httpx.get(url, headers={**HEADERS, "Accept": "text/html"}, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(r.text, "html.parser")

        rating = "N/A"
        rating_text = "N/A"

        # Look for the Zacks Rank
        rank_el = soup.select_one(".zr_rankbox .rank_view")
        if rank_el:
            rating = rank_el.get_text(strip=True)

        # Try to find rank description
        rank_desc = soup.select_one(".zr_rankbox .rank_chip")
        if rank_desc:
            rating_text = rank_desc.get_text(strip=True)

        # Extract key stats
        stats = {}
        stat_tables = soup.select(".key_stat_title")
        for st in stat_tables:
            key = st.get_text(strip=True)
            val_el = st.find_next_sibling()
            if val_el:
                stats[key] = val_el.get_text(strip=True)

        # Price target
        target = None
        target_el = soup.select_one("#price_target_summary")
        if target_el:
            target = target_el.get_text(strip=True)

        return {
            "symbol": symbol.upper(),
            "zacks_rank": rating,
            "rank_text": rating_text,
            "stats": stats,
            "price_target": target,
            "source": "zacks",
            "url": url
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol, "source": "zacks"}


# ═══════════════════════════════════════════════════════════════════════
#  OPENINSIDER  –  insider trading activity
# ═══════════════════════════════════════════════════════════════════════

def openinsider_latest(symbol: str = None) -> list:
    """Scrape latest insider trades from OpenInsider."""
    try:
        if symbol:
            url = f"http://openinsider.com/screener?s={symbol}&o=&pl=&ph=&ll=&lh=&fd=30&fdr=&td=0&tdr=&feession=&at=&a=&c=&cnt=25&page=1"
        else:
            url = "http://openinsider.com/latest-insider-trading"

        r = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(r.text, "html.parser")

        trades = []
        table = soup.select_one("table.tinytable")
        if not table:
            return trades

        rows = table.select("tbody tr")
        for row in rows[:30]:
            cells = row.find_all("td")
            if len(cells) < 12:
                continue
            try:
                trade = {
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
                    "source": "openinsider"
                }
                trades.append(trade)
            except (IndexError, AttributeError):
                continue
        return trades
    except Exception as e:
        return [{"error": str(e), "source": "openinsider"}]


def openinsider_top_buys() -> list:
    """Get top insider purchases (cluster buys)."""
    try:
        url = "http://openinsider.com/top-insider-purchases-of-the-month"
        r = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(r.text, "html.parser")
        trades = []
        table = soup.select_one("table.tinytable")
        if not table:
            return trades
        rows = table.select("tbody tr")
        for row in rows[:20]:
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
                })
            except (IndexError, AttributeError):
                continue
        return trades
    except Exception as e:
        return [{"error": str(e)}]


# ═══════════════════════════════════════════════════════════════════════
#  KALSHI  –  prediction markets
# ═══════════════════════════════════════════════════════════════════════

def kalshi_markets(query: str = None, category: str = "economics") -> list:
    """Fetch markets from Kalshi prediction exchange."""
    try:
        url = "https://api.elections.kalshi.com/trade-api/v2/markets"
        params = {"limit": 20, "status": "open"}
        if query:
            params["ticker"] = query.upper()

        r = httpx.get(url, headers={**HEADERS, "Accept": "application/json"}, timeout=10)
        data = r.json()
        markets = data.get("markets", [])
        results = []
        for m in markets[:20]:
            results.append({
                "ticker": m.get("ticker", ""),
                "title": m.get("title", ""),
                "subtitle": m.get("subtitle", ""),
                "yes_price": m.get("yes_bid", 0) / 100 if m.get("yes_bid") else None,
                "no_price": m.get("no_bid", 0) / 100 if m.get("no_bid") else None,
                "volume": m.get("volume", 0),
                "open_interest": m.get("open_interest", 0),
                "close_time": m.get("close_time", ""),
                "category": m.get("category", ""),
                "status": m.get("status", ""),
                "source": "kalshi"
            })
        return results
    except Exception as e:
        return [{"error": str(e), "source": "kalshi"}]


def kalshi_events(category: str = None) -> list:
    """Fetch event categories from Kalshi."""
    try:
        url = "https://api.elections.kalshi.com/trade-api/v2/events"
        params = {"limit": 25, "status": "open"}
        if category:
            params["series_ticker"] = category.upper()
        r = httpx.get(url, headers={**HEADERS, "Accept": "application/json"}, timeout=10)
        data = r.json()
        events = data.get("events", [])
        results = []
        for e in events[:25]:
            results.append({
                "ticker": e.get("event_ticker", ""),
                "title": e.get("title", ""),
                "category": e.get("category", ""),
                "markets_count": len(e.get("markets", [])),
                "source": "kalshi"
            })
        return results
    except Exception as e:
        return [{"error": str(e), "source": "kalshi"}]


# ═══════════════════════════════════════════════════════════════════════
#  NEWS AGGREGATOR  –  CNBC, Reuters, MarketWatch, Benzinga, etc.
# ═══════════════════════════════════════════════════════════════════════

def _parse_rss(url: str, source: str) -> list:
    """Generic RSS feed parser."""
    try:
        r = httpx.get(url, headers=HEADERS, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(r.text, "xml")
        items = soup.find_all("item")
        articles = []
        for item in items[:15]:
            title = item.find("title")
            link = item.find("link")
            pub = item.find("pubDate")
            desc = item.find("description")
            articles.append({
                "title": title.get_text(strip=True) if title else "",
                "url": link.get_text(strip=True) if link else "",
                "published": pub.get_text(strip=True) if pub else "",
                "summary": _clean_html(desc.get_text(strip=True)) if desc else "",
                "source": source
            })
        return articles
    except Exception as e:
        return [{"error": str(e), "source": source}]


def _clean_html(text: str) -> str:
    """Strip HTML tags from RSS descriptions."""
    clean = re.sub(r'<[^>]+>', '', text)
    return clean[:300] if len(clean) > 300 else clean


def news_cnbc() -> list:
    """CNBC top financial news via RSS."""
    return _parse_rss("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "CNBC")


def news_cnbc_markets() -> list:
    """CNBC market-specific news."""
    return _parse_rss("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258", "CNBC Markets")


def news_reuters_markets() -> list:
    """Reuters business/markets news via RSS."""
    return _parse_rss("https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best", "Reuters")


def news_marketwatch() -> list:
    """MarketWatch top stories via RSS."""
    return _parse_rss("https://feeds.marketwatch.com/marketwatch/topstories/", "MarketWatch")


def news_marketwatch_markets() -> list:
    """MarketWatch market pulse."""
    return _parse_rss("https://feeds.marketwatch.com/marketwatch/marketpulse/", "MarketWatch Pulse")


def news_benzinga() -> list:
    """Benzinga news feed."""
    return _parse_rss("https://www.benzinga.com/feed", "Benzinga")


def news_wsj_markets() -> list:
    """WSJ Markets RSS."""
    return _parse_rss("https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "WSJ")


def news_yahoo_finance() -> list:
    """Yahoo Finance RSS."""
    return _parse_rss("https://finance.yahoo.com/news/rssindex", "Yahoo Finance")


def news_seeking_alpha() -> list:
    """Seeking Alpha market news."""
    return _parse_rss("https://seekingalpha.com/market_currents.xml", "Seeking Alpha")


def news_investing_com() -> list:
    """Investing.com news RSS."""
    return _parse_rss("https://www.investing.com/rss/news.rss", "Investing.com")


def news_for_symbol(symbol: str) -> list:
    """Get news for a specific stock symbol from Yahoo Finance."""
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        return _parse_rss(url, f"Yahoo ({symbol})")
    except Exception as e:
        return [{"error": str(e)}]


def aggregate_news(limit: int = 50) -> list:
    """Pull from all news sources and merge into a single feed sorted by recency."""
    all_news = []
    sources = [
        news_cnbc,
        news_cnbc_markets,
        news_marketwatch,
        news_wsj_markets,
        news_benzinga,
        news_yahoo_finance,
        news_seeking_alpha,
    ]

    threads = []
    results = [None] * len(sources)

    def fetch(idx, fn):
        try:
            results[idx] = fn()
        except Exception:
            results[idx] = []

    for i, fn in enumerate(sources):
        t = threading.Thread(target=fetch, args=(i, fn))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=12)

    for r in results:
        if r:
            all_news.extend([a for a in r if "error" not in a])

    # Deduplicate by title
    seen = set()
    unique = []
    for article in all_news:
        key = article.get("title", "").lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(article)

    return unique[:limit]


# ═══════════════════════════════════════════════════════════════════════
#  MARKET OVERVIEW  –  indices, futures, crypto
# ═══════════════════════════════════════════════════════════════════════

def market_overview() -> dict:
    """Get major indices, futures, crypto overview."""
    symbols = {
        "indices": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
        "futures": ["ES=F", "NQ=F", "YM=F", "CL=F", "GC=F"],
        "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"]
    }
    result = {}
    for category, syms in symbols.items():
        result[category] = []
        for sym in syms:
            q = yahoo_quote(sym)
            if "error" not in q:
                name_map = {
                    "^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "Nasdaq",
                    "^RUT": "Russell 2000", "^VIX": "VIX",
                    "ES=F": "S&P Futures", "NQ=F": "Nasdaq Futures",
                    "YM=F": "Dow Futures", "CL=F": "Crude Oil", "GC=F": "Gold",
                    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"
                }
                q["name"] = name_map.get(sym, sym)
                result[category].append(q)
    return result


# ═══════════════════════════════════════════════════════════════════════
#  FEAR & GREED / ECONOMIC CALENDAR helpers
# ═══════════════════════════════════════════════════════════════════════

def fear_greed_index() -> dict:
    """Fetch CNN Fear & Greed Index."""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = httpx.get(url, headers={**HEADERS, "Accept": "application/json"}, timeout=10)
        data = r.json()
        fg = data.get("fear_and_greed", {})
        return {
            "score": round(fg.get("score", 0), 1),
            "rating": fg.get("rating", "N/A"),
            "previous_close": round(fg.get("previous_close", 0), 1),
            "week_ago": round(fg.get("previous_1_week", 0), 1),
            "month_ago": round(fg.get("previous_1_month", 0), 1),
            "year_ago": round(fg.get("previous_1_year", 0), 1),
            "source": "CNN Fear & Greed"
        }
    except Exception as e:
        return {"error": str(e), "source": "CNN Fear & Greed"}


# ═══════════════════════════════════════════════════════════════════════
#  CONVENIENCE: Full research bundle for a single ticker
# ═══════════════════════════════════════════════════════════════════════

def full_research(symbol: str) -> dict:
    """Bundle quote + fundamentals + zacks + insider + news for one ticker."""
    results = {}
    threads = []

    def run(key, fn, *args):
        try:
            results[key] = fn(*args)
        except Exception as e:
            results[key] = {"error": str(e)}

    tasks = [
        ("quote", yahoo_quote, symbol),
        ("history", yahoo_history, symbol, "3mo", "1d"),
        ("fundamentals", yahoo_fundamentals, symbol),
        ("zacks", zacks_rating, symbol),
        ("insider_trades", openinsider_latest, symbol),
        ("news", news_for_symbol, symbol),
    ]

    for key, fn, *args in tasks:
        t = threading.Thread(target=run, args=(key, fn, *args))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=15)

    results["symbol"] = symbol.upper()
    return results
