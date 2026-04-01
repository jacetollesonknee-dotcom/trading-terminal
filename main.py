import os, json, time, random, threading, webbrowser
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
import anthropic
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

# Import the real data feeds and AI engine
import data_feeds
from ai_engine import AIAnalysisEngine

SCHWAB_API_KEY = os.getenv("SCHWAB_API_KEY", "")
SCHWAB_SECRET = os.getenv("SCHWAB_SECRET", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PAPER_TRADING = True
DEMO_MODE = SCHWAB_API_KEY in ("", "pending") or SCHWAB_SECRET in ("", "pending")
USE_LIVE_DATA = os.getenv("USE_LIVE_DATA", "true").lower() == "true"

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

portfolio = {"cash": 30000.00, "positions": {}, "trade_log": []}
chat_history = []
ai_engine = AIAnalysisEngine(api_key=ANTHROPIC_API_KEY)

# Fallback demo prices if live data fails
PRICES = {
    "AAPL": 213.50, "NVDA": 875.20, "SPY": 512.30, "TSLA": 248.60,
    "MSFT": 415.80, "AMZN": 198.40, "GOOGL": 165.20, "META": 523.10,
    "QQQ": 445.60, "AMD": 162.30, "PLTR": 24.80, "COIN": 238.50,
}

def demo_price(symbol):
    base = PRICES.get(symbol.upper(), 100.0)
    return round(base + base * random.uniform(-0.015, 0.015), 2)

def get_quote(symbol):
    """Try live Yahoo data first, fall back to demo."""
    symbol = symbol.upper()
    if USE_LIVE_DATA:
        live = data_feeds.yahoo_quote(symbol)
        if "error" not in live and live.get("price", 0) > 0:
            return live
    # Fallback to demo
    price = demo_price(symbol)
    change = round(random.uniform(-3, 3), 2)
    return {
        "symbol": symbol, "price": price, "change": change,
        "change_pct": round((change / price) * 100, 2),
        "volume": random.randint(10000000, 80000000),
        "bid": round(price - 0.02, 2), "ask": round(price + 0.02, 2),
        "high": round(price * 1.012, 2), "low": round(price * 0.988, 2),
        "open": round(price * 0.998, 2), "close": round(price * 1.001, 2),
        "demo": True
    }

def get_history(symbol, days=30):
    """Try live Yahoo history first, fall back to demo."""
    symbol = symbol.upper()
    if USE_LIVE_DATA:
        period_map = {7: "5d", 14: "2wk", 30: "1mo", 60: "3mo", 90: "3mo",
                      180: "6mo", 365: "1y"}
        period = period_map.get(days, "3mo")
        live = data_feeds.yahoo_history(symbol, period=period)
        if live and "error" not in (live[0] if live else {}):
            return live

    # Fallback to demo
    base = PRICES.get(symbol, 100.0)
    price = base * 0.92
    result = []
    for i in range(days):
        price = round(price * random.uniform(0.988, 1.012), 2)
        date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        result.append({
            "date": date, "open": price,
            "high": round(price * random.uniform(1.002, 1.015), 2),
            "low": round(price * random.uniform(0.985, 0.998), 2),
            "close": price, "volume": random.randint(5000000, 50000000)
        })
    return result

def get_options(symbol):
    symbol = symbol.upper()
    price = get_quote(symbol).get("price", demo_price(symbol))
    calls, puts = [], []
    exps = [(datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d") for d in [7, 14, 30, 60, 90]]
    strikes = [round(price * m, 1) for m in [0.90, 0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05, 1.10]]
    for exp in exps:
        dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
        for strike in strikes:
            itm_c = strike < price
            iv = round(random.uniform(0.20, 0.55), 3)
            d_c = round(random.uniform(0.4, 0.9) if itm_c else random.uniform(0.1, 0.45), 3)
            base = {
                "expiration": exp, "strike": strike, "dte": dte, "iv": iv,
                "gamma": round(random.uniform(0.01, 0.08), 4),
                "theta": round(-random.uniform(0.02, 0.15), 3),
                "vega": round(random.uniform(0.05, 0.40), 3),
                "volume": random.randint(0, 5000),
                "open_interest": random.randint(100, 20000)
            }
            calls.append({
                **base, "side": "CALL", "itm": itm_c, "delta": d_c,
                "bid": round(max(0, price - strike + random.uniform(0, 2)), 2),
                "ask": round(max(0, price - strike + random.uniform(0.1, 2.5)), 2),
                "last": round(max(0, price - strike + random.uniform(0, 2)), 2)
            })
            puts.append({
                **base, "side": "PUT", "itm": not itm_c, "delta": round(-(1 - d_c), 3),
                "bid": round(max(0, strike - price + random.uniform(0, 2)), 2),
                "ask": round(max(0, strike - price + random.uniform(0.1, 2.5)), 2),
                "last": round(max(0, strike - price + random.uniform(0, 2)), 2)
            })
    return {"symbol": symbol, "underlying_price": price, "calls": calls, "puts": puts, "demo": True}

def get_signals(symbol):
    history = get_history(symbol, days=60)
    closes = [c["close"] for c in history if "close" in c]
    if len(closes) < 21:
        return {"symbol": symbol, "signal": "INSUFFICIENT DATA", "confidence": "LOW"}
    sma10 = sum(closes[-10:]) / 10
    sma20 = sum(closes[-20:]) / 20
    gains, losses = [], []
    for i in range(1, min(15, len(closes))):
        diff = closes[-i] - closes[-(i + 1)]
        (gains if diff > 0 else losses).append(abs(diff))
    avg_gain = sum(gains) / 14 if gains else 0
    avg_loss = sum(losses) / 14 if losses else 0.001
    rsi = round(100 - (100 / (1 + avg_gain / avg_loss)), 2)
    highs = [c["high"] for c in history[-15:]]
    lows = [c["low"] for c in history[-15:]]
    n = min(len(highs), len(lows), 14)
    atr = sum([highs[i] - lows[i] for i in range(1, n)]) / max(n - 1, 1)
    price = closes[-1]
    stop_loss = round(price - 1.5 * atr, 2)
    if sma10 > sma20 and rsi < 70:
        signal, confidence = "BUY", ("HIGH" if rsi < 55 else "MEDIUM")
    elif sma10 < sma20 and rsi > 60:
        signal, confidence = "SELL", ("HIGH" if rsi > 70 else "MEDIUM")
    else:
        signal, confidence = "HOLD", "LOW"
    return {
        "symbol": symbol, "signal": signal, "confidence": confidence,
        "rsi": rsi, "sma10": round(sma10, 2), "sma20": round(sma20, 2),
        "atr": round(atr, 2), "stop_loss": stop_loss, "price": round(price, 2)
    }

def get_summary():
    positions = portfolio["positions"]
    market_value = sum(p["current_price"] * p["qty"] for p in positions.values())
    total_pnl = sum(p["pnl"] for p in positions.values())
    return {
        "cash": round(portfolio["cash"], 2),
        "market_value": round(market_value, 2),
        "total_value": round(portfolio["cash"] + market_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round((total_pnl / 30000) * 100, 2),
        "positions": positions,
        "trade_log": portfolio["trade_log"][-20:]
    }

def do_buy(symbol, qty, order_type="MARKET", limit_price=None):
    quote = get_quote(symbol)
    price = limit_price if (order_type == "LIMIT" and limit_price) else quote["price"]
    cost = price * qty
    if cost > portfolio["cash"]:
        return {"error": "Insufficient cash. Need $" + str(round(cost, 2))}
    portfolio["cash"] -= cost
    pos = portfolio["positions"]
    if symbol in pos:
        total = pos[symbol]["qty"] + qty
        pos[symbol]["avg_cost"] = round(((pos[symbol]["avg_cost"] * pos[symbol]["qty"]) + cost) / total, 4)
        pos[symbol]["qty"] = total
    else:
        pos[symbol] = {"qty": qty, "avg_cost": price, "current_price": price, "pnl": 0, "pnl_pct": 0}
    trade = {"action": "BUY", "symbol": symbol, "qty": qty, "price": price,
             "total": cost, "time": datetime.now().strftime("%H:%M %m/%d"), "type": order_type}
    portfolio["trade_log"].append(trade)
    return {"success": True, "trade": trade, "cash_remaining": round(portfolio["cash"], 2)}

def do_sell(symbol, qty, order_type="MARKET", limit_price=None):
    pos = portfolio["positions"]
    if symbol not in pos or pos[symbol]["qty"] < qty:
        return {"error": "Cannot sell " + str(qty) + " of " + symbol}
    quote = get_quote(symbol)
    price = limit_price if (order_type == "LIMIT" and limit_price) else quote["price"]
    pnl = (price - pos[symbol]["avg_cost"]) * qty
    portfolio["cash"] += price * qty
    pos[symbol]["qty"] -= qty
    if pos[symbol]["qty"] == 0:
        del pos[symbol]
    trade = {"action": "SELL", "symbol": symbol, "qty": qty, "price": price,
             "total": round(price * qty, 2), "pnl": round(pnl, 2),
             "time": datetime.now().strftime("%H:%M %m/%d"), "type": order_type}
    portfolio["trade_log"].append(trade)
    return {"success": True, "trade": trade, "realized_pnl": round(pnl, 2)}

def update_prices():
    while True:
        for symbol in list(portfolio["positions"].keys()):
            q = get_quote(symbol)
            if "price" in q:
                pos = portfolio["positions"][symbol]
                pos["current_price"] = q["price"]
                pos["pnl"] = round((q["price"] - pos["avg_cost"]) * pos["qty"], 2)
                pos["pnl_pct"] = round(((q["price"] - pos["avg_cost"]) / pos["avg_cost"]) * 100, 2)
        socketio.emit("portfolio_update", get_summary())
        time.sleep(15)

SYSTEM_PROMPT = """You are an elite trading research assistant with 50 years of Wall Street experience.
You help a trader with a $30,000 paper trading account analyze stocks and options.

DATA SOURCES AVAILABLE:
- Yahoo Finance (live quotes, charts, fundamentals)
- Zacks Research (analyst ratings, price targets)
- OpenInsider (insider buy/sell activity)
- Kalshi (prediction market probabilities)
- Multi-source news (CNBC, MarketWatch, WSJ, Benzinga, Seeking Alpha, Reuters)
- CNN Fear & Greed Index
- Technical analysis (RSI, SMA, ATR, signals)

GUIDELINES:
- Give concise actionable research with bullet points
- Always include stop loss levels on buy recommendations
- Reference RSI SMA ATR signals when relevant
- Cross-reference insider activity and analyst ratings when available
- Explain options Greeks clearly when discussing options
- Reference relevant news and prediction market sentiment
- Frame everything as research not licensed financial advice
- Remind the user this is paper trading with no real money at risk
"""

def ai_chat(user_message, context=None):
    context_str = ("\n\n[MARKET CONTEXT]\n" + json.dumps(context, indent=2)) if context else ""
    chat_history.append({"role": "user", "content": user_message + context_str})
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5-20250514", max_tokens=2048,
        system=SYSTEM_PROMPT, messages=chat_history[-20:]
    )
    reply = response.content[0].text
    chat_history.append({"role": "assistant", "content": reply})
    return reply

# ═══════════════════════════════════════════════════════════════════════
#  ROUTES  –  Original endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    return jsonify({"demo_mode": DEMO_MODE, "paper_trading": PAPER_TRADING,
                    "live_data": USE_LIVE_DATA})

@app.route("/api/quote/<symbol>")
def api_quote(symbol):
    return jsonify(get_quote(symbol))

@app.route("/api/options/<symbol>")
def api_options(symbol):
    return jsonify(get_options(symbol))

@app.route("/api/history/<symbol>")
def api_history(symbol):
    days = int(request.args.get("days", 30))
    return jsonify(get_history(symbol, days))

@app.route("/api/signals/<symbol>")
def api_signals(symbol):
    return jsonify(get_signals(symbol))

@app.route("/api/portfolio")
def api_portfolio():
    return jsonify(get_summary())

@app.route("/api/trade", methods=["POST"])
def api_trade():
    data = request.json
    action = data.get("action", "").upper()
    symbol = data.get("symbol", "").upper()
    qty = int(data.get("qty", 1))
    otype = data.get("order_type", "MARKET")
    limit = data.get("limit_price")
    if action == "BUY":
        result = do_buy(symbol, qty, otype, limit)
    elif action == "SELL":
        result = do_sell(symbol, qty, otype, limit)
    else:
        result = {"error": "Invalid action"}
    return jsonify(result)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    message = data.get("message", "")
    symbol = data.get("symbol")
    context = None
    if symbol:
        context = {
            "quote": get_quote(symbol),
            "signals": get_signals(symbol),
            "portfolio": get_summary()
        }
    return jsonify({"reply": ai_chat(message, context)})

# ═══════════════════════════════════════════════════════════════════════
#  NEW ROUTES  –  Live data feed endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/news")
def api_news():
    """Aggregated news from all sources."""
    limit = int(request.args.get("limit", 40))
    return jsonify(data_feeds.aggregate_news(limit=limit))

@app.route("/api/news/<symbol>")
def api_news_symbol(symbol):
    """News for a specific ticker."""
    return jsonify(data_feeds.news_for_symbol(symbol))

@app.route("/api/insider")
def api_insider():
    """Latest insider trades across all stocks."""
    return jsonify(data_feeds.openinsider_latest())

@app.route("/api/insider/<symbol>")
def api_insider_symbol(symbol):
    """Insider trades for a specific symbol."""
    return jsonify(data_feeds.openinsider_latest(symbol))

@app.route("/api/insider/top-buys")
def api_insider_top():
    """Top insider purchases this month."""
    return jsonify(data_feeds.openinsider_top_buys())

@app.route("/api/zacks/<symbol>")
def api_zacks(symbol):
    """Zacks rating and research for a symbol."""
    return jsonify(data_feeds.zacks_rating(symbol))

@app.route("/api/kalshi")
def api_kalshi():
    """Kalshi prediction markets."""
    query = request.args.get("q")
    return jsonify(data_feeds.kalshi_markets(query))

@app.route("/api/kalshi/events")
def api_kalshi_events():
    """Kalshi event categories."""
    category = request.args.get("category")
    return jsonify(data_feeds.kalshi_events(category))

@app.route("/api/market-overview")
def api_market_overview():
    """Major indices, futures, crypto."""
    return jsonify(data_feeds.market_overview())

@app.route("/api/movers")
def api_movers():
    """Market movers – gainers, losers, most active."""
    return jsonify(data_feeds.yahoo_movers())

@app.route("/api/trending")
def api_trending():
    """Trending tickers on Yahoo Finance."""
    return jsonify(data_feeds.yahoo_trending())

@app.route("/api/fear-greed")
def api_fear_greed():
    """CNN Fear & Greed Index."""
    return jsonify(data_feeds.fear_greed_index())

@app.route("/api/fundamentals/<symbol>")
def api_fundamentals(symbol):
    """Yahoo Finance fundamentals."""
    return jsonify(data_feeds.yahoo_fundamentals(symbol))

@app.route("/api/research/<symbol>")
def api_research(symbol):
    """Full research bundle: quote + zacks + insider + news + signals."""
    research = data_feeds.full_research(symbol)
    research["signals"] = get_signals(symbol)
    return jsonify(research)

# ═══════════════════════════════════════════════════════════════════════
#  AI ENGINE ROUTES  –  Deep analysis & prediction
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/ai/full-analysis/<symbol>")
def api_ai_full(symbol):
    """Run the full 5-pass AI analysis pipeline (takes ~30-60s)."""
    result = ai_engine.full_analysis(symbol.upper())
    return jsonify(result)

@app.route("/api/ai/quick/<symbol>")
def api_ai_quick(symbol):
    """Quick single-pass AI analysis."""
    result = ai_engine.quick_analysis(symbol.upper())
    return jsonify({"symbol": symbol.upper(), "analysis": result})

@app.route("/api/ai/compare", methods=["POST"])
def api_ai_compare():
    """Compare multiple stocks."""
    data = request.json
    symbols = data.get("symbols", [])
    if not symbols:
        return jsonify({"error": "Provide symbols list"})
    result = ai_engine.compare_stocks(symbols)
    return jsonify({"symbols": symbols, "comparison": result})

@app.route("/api/ai/market-brief")
def api_ai_brief():
    """AI-generated morning market brief."""
    result = ai_engine.market_brief()
    return jsonify({"brief": result, "timestamp": datetime.now().isoformat()})

# ═══════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    live_str = "LIVE (Yahoo/Zacks/OpenInsider/Kalshi/News)" if USE_LIVE_DATA else "DEMO"
    print("\n" + "=" * 60)
    print("  AI TRADING RESEARCH TERMINAL")
    print("  Mode: PAPER TRADING")
    print(f"  Data: {live_str}")
    print("  Capital: $30,000.00")
    print("  " + "-" * 56)
    print("  Data Sources:")
    print("    Yahoo Finance   – quotes, charts, fundamentals")
    print("    Zacks            – analyst ratings & targets")
    print("    OpenInsider       – insider trading activity")
    print("    Kalshi            – prediction markets")
    print("    News              – CNBC, WSJ, MarketWatch, Benzinga,")
    print("                       Seeking Alpha, Reuters, Yahoo")
    print("    CNN Fear & Greed  – market sentiment index")
    print("=" * 60)
    if DEMO_MODE:
        print("\n  Schwab API pending – ready for connection.")
    print("  Opening browser at http://127.0.0.1:5000\n")
    threading.Thread(target=update_prices, daemon=True).start()
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    socketio.run(app, debug=False, port=5000)
