"""
main.py — AI Trading Research Terminal (Flask + SocketIO)
Powered by Claude Opus 4.7 with extended thinking + prompt caching.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_socketio import SocketIO

import data_feeds
from ai_engine import (
    AIAnalysisEngine,
    CHAT_SYSTEM,
    DEFAULT_MODEL,
    FAST_MODEL,
    OPUS,
    THINKING_BUDGET_QUICK,
)

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────
SCHWAB_API_KEY = os.getenv("SCHWAB_API_KEY", "")
SCHWAB_SECRET = os.getenv("SCHWAB_SECRET", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
DEMO_MODE = SCHWAB_API_KEY in ("", "pending") or SCHWAB_SECRET in ("", "pending")
USE_LIVE_DATA = os.getenv("USE_LIVE_DATA", "true").lower() == "true"
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "30000"))
STATE_PATH = Path(os.getenv("STATE_PATH", "trader_state.json"))
PORT = int(os.getenv("PORT", "5000"))

# ── App ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(32)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
ai_engine = AIAnalysisEngine(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ── State ────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
state = {
    "cash": INITIAL_CASH,
    "initial_cash": INITIAL_CASH,
    "positions": {},   # symbol -> {qty, avg_cost, current_price, pnl, pnl_pct}
    "trade_log": [],
    "watchlist": ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "MSFT"],
    "alerts": [],      # {id, symbol, type, level, created, triggered}
}
chat_sessions: dict = {}  # session_id -> [{role, content}, ...]


def _save_state():
    try:
        with _state_lock:
            STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
    except Exception as e:  # noqa: BLE001
        print(f"[state] save failed: {e}")


def _load_state():
    if not STATE_PATH.exists():
        return
    try:
        with _state_lock:
            saved = json.loads(STATE_PATH.read_text() or "{}")
            for k, v in saved.items():
                state[k] = v
        print(f"[state] loaded from {STATE_PATH}")
    except Exception as e:  # noqa: BLE001
        print(f"[state] load failed: {e}")


# ── Demo fallback prices ─────────────────────────────────────────────────
PRICES = {
    "AAPL": 213.50, "NVDA": 875.20, "SPY": 512.30, "TSLA": 248.60,
    "MSFT": 415.80, "AMZN": 198.40, "GOOGL": 165.20, "META": 523.10,
    "QQQ": 445.60, "AMD": 162.30, "PLTR": 24.80, "COIN": 238.50,
}


def _demo_price(symbol: str) -> float:
    base = PRICES.get(symbol.upper(), 100.0)
    return round(base + base * random.uniform(-0.015, 0.015), 2)


def get_quote(symbol: str) -> dict:
    sym = symbol.upper()
    if USE_LIVE_DATA:
        live = data_feeds.yahoo_quote(sym)
        if "error" not in live and (live.get("price") or 0) > 0:
            return live
    price = _demo_price(sym)
    change = round(random.uniform(-3, 3), 2)
    return {
        "symbol": sym, "price": price, "change": change,
        "change_pct": round((change / price) * 100, 2),
        "volume": random.randint(10_000_000, 80_000_000),
        "bid": round(price - 0.02, 2), "ask": round(price + 0.02, 2),
        "high": round(price * 1.012, 2), "low": round(price * 0.988, 2),
        "open": round(price * 0.998, 2), "close": round(price * 1.001, 2),
        "demo": True,
    }


def get_history(symbol: str, days: int = 90) -> list:
    sym = symbol.upper()
    if USE_LIVE_DATA:
        period_map = {7: "5d", 14: "1mo", 30: "1mo", 60: "3mo", 90: "3mo",
                      180: "6mo", 365: "1y", 730: "2y", 1825: "5y"}
        period = period_map.get(days, "3mo")
        live = data_feeds.yahoo_history(sym, period=period)
        if live and "error" not in (live[0] if live else {}):
            return live
    base = PRICES.get(sym, 100.0)
    price = base * 0.92
    out = []
    for i in range(days):
        price = round(price * random.uniform(0.988, 1.012), 2)
        date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        out.append({"date": date, "open": price,
                    "high": round(price * random.uniform(1.002, 1.015), 2),
                    "low": round(price * random.uniform(0.985, 0.998), 2),
                    "close": price, "volume": random.randint(5_000_000, 50_000_000)})
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Technical indicators (real implementations, vector friendly)
# ─────────────────────────────────────────────────────────────────────────

def _sma(values: list, period: int) -> list:
    out = [None] * len(values)
    if len(values) < period:
        return out
    s = sum(values[:period])
    out[period - 1] = s / period
    for i in range(period, len(values)):
        s += values[i] - values[i - period]
        out[i] = s / period
    return out


def _ema(values: list, period: int) -> list:
    out = [None] * len(values)
    if len(values) < period:
        return out
    k = 2 / (period + 1)
    sma = sum(values[:period]) / period
    out[period - 1] = sma
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes: list, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _atr(highs: list, lows: list, closes: list, period: int = 14) -> float | None:
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None
    trs = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return round(atr, 4)


def _bollinger(closes: list, period: int = 20, k: float = 2.0):
    if len(closes) < period:
        return None, None, None
    recent = closes[-period:]
    mean = sum(recent) / period
    var = sum((x - mean) ** 2 for x in recent) / period
    sd = var ** 0.5
    return round(mean - k * sd, 2), round(mean, 2), round(mean + k * sd, 2)


def get_signals(symbol: str) -> dict:
    history = get_history(symbol, days=250)
    closes = [c["close"] for c in history if "close" in c]
    highs = [c["high"] for c in history if "high" in c]
    lows = [c["low"] for c in history if "low" in c]

    if len(closes) < 21:
        return {"symbol": symbol, "signal": "INSUFFICIENT DATA",
                "confidence": "LOW", "price": closes[-1] if closes else None}

    sma10 = _sma(closes, 10)[-1] or 0
    sma20 = _sma(closes, 20)[-1] or 0
    sma50 = _sma(closes, 50)[-1] if len(closes) >= 50 else None
    sma200 = _sma(closes, 200)[-1] if len(closes) >= 200 else None

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd = None
    macd_signal = None
    if ema12[-1] is not None and ema26[-1] is not None:
        macd_line = [ (a - b) if (a is not None and b is not None) else None
                      for a, b in zip(ema12, ema26) ]
        macd = round(macd_line[-1], 4)
        macd_clean = [m for m in macd_line if m is not None]
        sig_arr = _ema(macd_clean, 9)
        if sig_arr[-1] is not None:
            macd_signal = round(sig_arr[-1], 4)

    rsi = _rsi(closes, 14)
    atr = _atr(highs, lows, closes, 14)
    bb_low, bb_mid, bb_up = _bollinger(closes, 20, 2.0)
    price = closes[-1]
    stop_loss = round(price - 1.5 * (atr or price * 0.02), 2)

    # Composite score
    score = 0
    if rsi is not None:
        if rsi < 35: score += 2
        elif rsi > 70: score -= 2
        elif rsi < 50: score += 1
        else: score -= 1
    if sma10 > sma20: score += 1
    else: score -= 1
    if sma50 and price > sma50: score += 1
    if sma200 and price > sma200: score += 2
    if macd is not None and macd_signal is not None:
        if macd > macd_signal: score += 1
        else: score -= 1

    if score >= 3:
        signal, confidence = "BUY", "HIGH" if score >= 5 else "MEDIUM"
    elif score <= -3:
        signal, confidence = "SELL", "HIGH" if score <= -5 else "MEDIUM"
    else:
        signal, confidence = "HOLD", "LOW"

    return {
        "symbol": symbol.upper(),
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "rsi": rsi,
        "sma10": round(sma10, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2) if sma50 else None,
        "sma200": round(sma200, 2) if sma200 else None,
        "macd": macd,
        "macd_signal": macd_signal,
        "atr": atr,
        "bb_lower": bb_low, "bb_middle": bb_mid, "bb_upper": bb_up,
        "stop_loss": stop_loss,
        "price": round(price, 2),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Options (synthetic, since no real chain feed)
# ─────────────────────────────────────────────────────────────────────────

def get_options(symbol: str) -> dict:
    sym = symbol.upper()
    price = get_quote(sym).get("price", _demo_price(sym))
    calls, puts = [], []
    exps = [(datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            for d in [7, 14, 30, 60, 90]]
    strikes = [round(price * m, 1) for m in
               [0.85, 0.90, 0.95, 0.97, 0.99, 1.00, 1.01, 1.03, 1.05, 1.10, 1.15]]
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
                "vega":  round(random.uniform(0.05, 0.40), 3),
                "volume": random.randint(0, 5000),
                "open_interest": random.randint(100, 20000),
            }
            calls.append({**base, "side": "CALL", "itm": itm_c, "delta": d_c,
                          "bid":  round(max(0, price - strike + random.uniform(0, 2)), 2),
                          "ask":  round(max(0, price - strike + random.uniform(0.1, 2.5)), 2),
                          "last": round(max(0, price - strike + random.uniform(0, 2)), 2)})
            puts.append({**base, "side": "PUT", "itm": not itm_c,
                         "delta": round(-(1 - d_c), 3),
                         "bid":  round(max(0, strike - price + random.uniform(0, 2)), 2),
                         "ask":  round(max(0, strike - price + random.uniform(0.1, 2.5)), 2),
                         "last": round(max(0, strike - price + random.uniform(0, 2)), 2)})
    return {"symbol": sym, "underlying_price": price,
            "calls": calls, "puts": puts, "demo": True}


# ─────────────────────────────────────────────────────────────────────────
#  Portfolio
# ─────────────────────────────────────────────────────────────────────────

def get_summary() -> dict:
    positions = state["positions"]
    market_value = sum(p["current_price"] * p["qty"] for p in positions.values())
    total_pnl = sum(p["pnl"] for p in positions.values())
    total_value = state["cash"] + market_value
    initial = state.get("initial_cash", INITIAL_CASH) or 1
    return {
        "cash": round(state["cash"], 2),
        "initial_cash": round(initial, 2),
        "market_value": round(market_value, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(((total_value - initial) / initial) * 100, 2),
        "positions": positions,
        "trade_log": state["trade_log"][-30:],
        "watchlist": state.get("watchlist", []),
    }


def do_buy(symbol: str, qty: int, order_type: str = "MARKET",
           limit_price: float | None = None) -> dict:
    quote = get_quote(symbol)
    price = limit_price if (order_type == "LIMIT" and limit_price) else quote["price"]
    cost = price * qty
    with _state_lock:
        if cost > state["cash"]:
            return {"error": f"Insufficient cash. Need ${cost:.2f}, have ${state['cash']:.2f}"}
        state["cash"] -= cost
        pos = state["positions"]
        if symbol in pos:
            total = pos[symbol]["qty"] + qty
            pos[symbol]["avg_cost"] = round(
                ((pos[symbol]["avg_cost"] * pos[symbol]["qty"]) + cost) / total, 4)
            pos[symbol]["qty"] = total
        else:
            pos[symbol] = {"qty": qty, "avg_cost": price, "current_price": price,
                           "pnl": 0, "pnl_pct": 0}
        trade = {"action": "BUY", "symbol": symbol, "qty": qty, "price": price,
                 "total": cost, "time": datetime.now().strftime("%H:%M %m/%d"),
                 "type": order_type}
        state["trade_log"].append(trade)
    _save_state()
    return {"success": True, "trade": trade,
            "cash_remaining": round(state["cash"], 2)}


def do_sell(symbol: str, qty: int, order_type: str = "MARKET",
            limit_price: float | None = None) -> dict:
    pos = state["positions"]
    with _state_lock:
        if symbol not in pos or pos[symbol]["qty"] < qty:
            return {"error": f"Cannot sell {qty} of {symbol}"}
        quote = get_quote(symbol)
        price = limit_price if (order_type == "LIMIT" and limit_price) else quote["price"]
        pnl = (price - pos[symbol]["avg_cost"]) * qty
        state["cash"] += price * qty
        pos[symbol]["qty"] -= qty
        if pos[symbol]["qty"] == 0:
            del pos[symbol]
        trade = {"action": "SELL", "symbol": symbol, "qty": qty, "price": price,
                 "total": round(price * qty, 2), "pnl": round(pnl, 2),
                 "time": datetime.now().strftime("%H:%M %m/%d"),
                 "type": order_type}
        state["trade_log"].append(trade)
    _save_state()
    return {"success": True, "trade": trade,
            "realized_pnl": round(pnl, 2)}


def update_prices_loop():
    """Background: update positions + watchlist quotes every 15s, push via SocketIO."""
    while True:
        try:
            with _state_lock:
                symbols = set(state["positions"].keys()) | set(state.get("watchlist", []))
            quotes = {}
            for sym in symbols:
                q = get_quote(sym)
                if "price" in q:
                    quotes[sym] = q
                    if sym in state["positions"]:
                        pos = state["positions"][sym]
                        pos["current_price"] = q["price"]
                        pos["pnl"] = round((q["price"] - pos["avg_cost"]) * pos["qty"], 2)
                        pos["pnl_pct"] = round(
                            ((q["price"] - pos["avg_cost"]) / pos["avg_cost"]) * 100, 2)
            socketio.emit("portfolio_update", get_summary())
            socketio.emit("quotes_update", quotes)
            _check_alerts(quotes)
        except Exception as e:  # noqa: BLE001
            print(f"[updater] {e}")
        time.sleep(15)


def _check_alerts(quotes: dict):
    triggered = []
    with _state_lock:
        for a in state.get("alerts", []):
            if a.get("triggered"):
                continue
            q = quotes.get(a["symbol"])
            if not q:
                continue
            price = q.get("price", 0)
            level = float(a["level"])
            if (a["type"] == "above" and price >= level) or \
               (a["type"] == "below" and price <= level):
                a["triggered"] = True
                a["triggered_at"] = datetime.now().isoformat()
                triggered.append(a)
    for a in triggered:
        socketio.emit("alert", a)


# ─────────────────────────────────────────────────────────────────────────
#  Chat (non-streaming + streaming SSE)
# ─────────────────────────────────────────────────────────────────────────

def _chat_history(session_id: str) -> list:
    return chat_sessions.setdefault(session_id, [])


def _build_context(symbol: str | None) -> dict | None:
    if not symbol:
        return None
    sym = symbol.upper()
    return {
        "quote": get_quote(sym),
        "signals": get_signals(sym),
        "recent_news": [n.get("title") for n in
                        data_feeds.news_for_symbol(sym)[:5] if isinstance(n, dict)],
        "portfolio": get_summary(),
    }


def _chat_session_id() -> str:
    sid = request.cookies.get("trader_sid") or request.headers.get("X-Session")
    return sid or "default"


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not ai_engine:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400
    data = request.json or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    symbol = data.get("symbol")
    context = _build_context(symbol)
    sid = _chat_session_id()
    history = _chat_history(sid)

    user_block = message
    if context:
        user_block += "\n\n[MARKET CONTEXT]\n" + json.dumps(context, indent=2, default=str)
    history.append({"role": "user", "content": user_block})

    try:
        reply = ai_engine._call_claude(  # noqa: SLF001
            CHAT_SYSTEM,
            user_block,
            model=DEFAULT_MODEL,
            thinking_budget=THINKING_BUDGET_QUICK,
            max_tokens=4096,
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500
    history.append({"role": "assistant", "content": reply})
    chat_sessions[sid] = history[-30:]
    return jsonify({"reply": reply, "model": DEFAULT_MODEL, "session": sid})


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """Server-Sent Events stream of the chat reply."""
    if not ai_engine:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400
    data = request.json or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    symbol = data.get("symbol")
    context = _build_context(symbol)
    sid = _chat_session_id()
    history = _chat_history(sid)

    user_block = message
    if context:
        user_block += "\n\n[MARKET CONTEXT]\n" + json.dumps(context, indent=2, default=str)

    @stream_with_context
    def gen():
        # Tell the client which model is responding
        yield f"event: meta\ndata: {json.dumps({'model': DEFAULT_MODEL, 'session': sid})}\n\n"
        full = []
        try:
            for chunk in ai_engine.stream_claude(
                CHAT_SYSTEM, user_block,
                model=DEFAULT_MODEL,
                max_tokens=4096,
                thinking_budget=THINKING_BUDGET_QUICK,
                history=history,
            ):
                full.append(chunk)
                yield f"data: {json.dumps({'delta': chunk})}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        # Persist history
        history.append({"role": "user", "content": user_block})
        history.append({"role": "assistant", "content": "".join(full)})
        chat_sessions[sid] = history[-30:]
        yield "event: done\ndata: {}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    resp = Response(gen(), headers=headers)
    if not request.cookies.get("trader_sid"):
        resp.set_cookie("trader_sid", sid, max_age=60 * 60 * 24 * 30, samesite="Lax")
    return resp


@app.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    sid = _chat_session_id()
    chat_sessions[sid] = []
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────────────────
#  Routes — market data
# ─────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "demo_mode": DEMO_MODE,
        "paper_trading": PAPER_TRADING,
        "live_data": USE_LIVE_DATA,
        "ai_ready": ai_engine is not None,
        "model": DEFAULT_MODEL,
        "fast_model": FAST_MODEL,
        "thinking_enabled": True,
        "initial_cash": state.get("initial_cash", INITIAL_CASH),
    })


# ── Brokers status (Phase 1.4a — read-only surface) ──────────────────────────
# Reads the engine's broker registry + keychain state. No OAuth flow yet;
# the "Connect" UI tells the operator to run `python -m cli connect <broker>`
# from a shell once their Schwab keys are ready.
@app.route("/api/brokers/status")
def api_brokers_status():
    """Per-broker enable flag + token-in-keychain check."""
    try:
        from config.secrets import get_schwab_token
        from config.settings import BrokerName, get_settings
        from ingestion.brokers.registry import list_registered

        settings = get_settings()
        rows = []
        for broker in list_registered():
            try:
                has_token = get_schwab_token(env="production") is not None
            except Exception:  # noqa: BLE001
                has_token = False
            rows.append({
                "name": broker.value,
                "enabled": settings.brokers_enabled.get(broker.value, False),
                "has_token": has_token,
                "env": "production",
            })
        return jsonify({"brokers": rows, "engine_available": True})
    except Exception as e:  # noqa: BLE001
        # Engine import failed (e.g. running terminal-only). Surface gracefully.
        return jsonify({
            "brokers": [
                {"name": "schwab", "enabled": False, "has_token": False, "env": "production"},
                {"name": "tos", "enabled": False, "has_token": False, "env": "production"},
            ],
            "engine_available": False,
            "error": str(e),
        })


@app.route("/api/quote/<symbol>")
def api_quote(symbol):
    return jsonify(get_quote(symbol))


@app.route("/api/quotes")
def api_quotes_batch():
    syms = [s.strip().upper() for s in
            (request.args.get("symbols") or "").split(",") if s.strip()]
    return jsonify({s: get_quote(s) for s in syms})


@app.route("/api/options/<symbol>")
def api_options(symbol):
    return jsonify(get_options(symbol))


@app.route("/api/history/<symbol>")
def api_history(symbol):
    days = int(request.args.get("days", 90))
    return jsonify(get_history(symbol, days))


@app.route("/api/signals/<symbol>")
def api_signals(symbol):
    return jsonify(get_signals(symbol))


@app.route("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify([])
    return jsonify(data_feeds.yahoo_search(q))


# ── Portfolio + trading ──
@app.route("/api/portfolio")
def api_portfolio():
    return jsonify(get_summary())


@app.route("/api/portfolio/reset", methods=["POST"])
def api_portfolio_reset():
    with _state_lock:
        state["cash"] = INITIAL_CASH
        state["initial_cash"] = INITIAL_CASH
        state["positions"] = {}
        state["trade_log"] = []
    _save_state()
    return jsonify({"ok": True, "summary": get_summary()})


@app.route("/api/trade", methods=["POST"])
def api_trade():
    data = request.json or {}
    action = (data.get("action") or "").upper()
    symbol = (data.get("symbol") or "").upper()
    try:
        qty = int(data.get("qty", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid qty"}), 400
    otype = (data.get("order_type") or "MARKET").upper()
    limit = data.get("limit_price")
    try:
        limit = float(limit) if limit not in (None, "") else None
    except (TypeError, ValueError):
        limit = None

    if not symbol or qty <= 0:
        return jsonify({"error": "symbol and positive qty required"}), 400
    if action == "BUY":
        result = do_buy(symbol, qty, otype, limit)
    elif action == "SELL":
        result = do_sell(symbol, qty, otype, limit)
    else:
        result = {"error": "Invalid action"}
    return jsonify(result)


# ── Watchlist ──
@app.route("/api/watchlist", methods=["GET", "POST", "DELETE"])
def api_watchlist():
    if request.method == "GET":
        return jsonify({"watchlist": state.get("watchlist", [])})
    body = request.json or {}
    sym = (body.get("symbol") or "").upper().strip()
    if not sym:
        return jsonify({"error": "symbol required"}), 400
    with _state_lock:
        wl = state.setdefault("watchlist", [])
        if request.method == "POST":
            if sym not in wl:
                wl.insert(0, sym)
                wl[:] = wl[:25]
        else:  # DELETE
            if sym in wl:
                wl.remove(sym)
    _save_state()
    return jsonify({"watchlist": state["watchlist"]})


@app.route("/api/watchlist/quotes")
def api_watchlist_quotes():
    syms = state.get("watchlist", [])
    return jsonify({s: get_quote(s) for s in syms})


# ── Alerts ──
@app.route("/api/alerts", methods=["GET", "POST", "DELETE"])
def api_alerts():
    if request.method == "GET":
        return jsonify(state.get("alerts", []))
    body = request.json or {}
    if request.method == "DELETE":
        aid = body.get("id")
        with _state_lock:
            state["alerts"] = [a for a in state.get("alerts", []) if a.get("id") != aid]
        _save_state()
        return jsonify({"ok": True})
    sym = (body.get("symbol") or "").upper()
    typ = body.get("type", "above")
    try:
        level = float(body.get("level"))
    except (TypeError, ValueError):
        return jsonify({"error": "level must be a number"}), 400
    if typ not in ("above", "below"):
        return jsonify({"error": "type must be 'above' or 'below'"}), 400
    alert = {"id": str(uuid.uuid4())[:8], "symbol": sym, "type": typ,
             "level": level, "created": datetime.now().isoformat(),
             "triggered": False}
    with _state_lock:
        state.setdefault("alerts", []).append(alert)
    _save_state()
    return jsonify(alert)


# ── News / insider / Kalshi / fundamentals ──
@app.route("/api/news")
def api_news():
    limit = int(request.args.get("limit", 40))
    return jsonify(data_feeds.aggregate_news(limit=limit))


@app.route("/api/news/<symbol>")
def api_news_symbol(symbol):
    return jsonify(data_feeds.news_for_symbol(symbol))


@app.route("/api/insider")
def api_insider():
    return jsonify(data_feeds.openinsider_latest())


@app.route("/api/insider/<symbol>")
def api_insider_symbol(symbol):
    return jsonify(data_feeds.openinsider_latest(symbol))


@app.route("/api/insider/top-buys")
def api_insider_top():
    return jsonify(data_feeds.openinsider_top_buys())


@app.route("/api/zacks/<symbol>")
def api_zacks(symbol):
    return jsonify(data_feeds.zacks_rating(symbol))


@app.route("/api/kalshi")
def api_kalshi():
    query = request.args.get("q")
    return jsonify(data_feeds.kalshi_markets(query))


@app.route("/api/kalshi/events")
def api_kalshi_events():
    category = request.args.get("category")
    return jsonify(data_feeds.kalshi_events(category))


@app.route("/api/market-overview")
def api_market_overview():
    return jsonify(data_feeds.market_overview())


@app.route("/api/movers")
def api_movers():
    return jsonify(data_feeds.yahoo_movers())


@app.route("/api/trending")
def api_trending():
    return jsonify(data_feeds.yahoo_trending())


@app.route("/api/fear-greed")
def api_fear_greed():
    return jsonify(data_feeds.fear_greed_index())


@app.route("/api/fundamentals/<symbol>")
def api_fundamentals(symbol):
    return jsonify(data_feeds.yahoo_fundamentals(symbol))


@app.route("/api/research/<symbol>")
def api_research(symbol):
    research = data_feeds.full_research(symbol)
    research["signals"] = get_signals(symbol)
    return jsonify(research)


# ── AI engine ──
@app.route("/api/ai/full-analysis/<symbol>")
def api_ai_full(symbol):
    if not ai_engine:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400
    return jsonify(ai_engine.full_analysis(symbol.upper()))


@app.route("/api/ai/quick/<symbol>")
def api_ai_quick(symbol):
    if not ai_engine:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400
    result = ai_engine.quick_analysis(symbol.upper())
    return jsonify({"symbol": symbol.upper(), "analysis": result, "model": DEFAULT_MODEL})


@app.route("/api/ai/compare", methods=["POST"])
def api_ai_compare():
    if not ai_engine:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400
    data = request.json or {}
    symbols = data.get("symbols", [])
    if not symbols:
        return jsonify({"error": "Provide symbols list"}), 400
    result = ai_engine.compare_stocks(symbols)
    return jsonify({"symbols": symbols, "comparison": result, "model": DEFAULT_MODEL})


@app.route("/api/ai/market-brief")
def api_ai_brief():
    if not ai_engine:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400
    result = ai_engine.market_brief()
    return jsonify({"brief": result, "model": DEFAULT_MODEL,
                    "timestamp": datetime.now().isoformat()})


@app.route("/api/ai/full-analysis/<symbol>/stream")
def api_ai_full_stream(symbol):
    """Stream the full 5-pass analysis as it happens (SSE)."""
    if not ai_engine:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400
    sym = symbol.upper()

    @stream_with_context
    def gen():
        yield f"event: stage\ndata: {json.dumps({'stage': 'gather', 'message': 'Gathering all live data...'})}\n\n"
        try:
            data = ai_engine.gather_data(sym)
            yield f"event: stage\ndata: {json.dumps({'stage': 'gathered', 'sources': [k for k in data if k not in ('symbol','timestamp')]})}\n\n"

            stages = [
                ("technical", "Technical Analyst", ai_engine.run_technical_analysis),
                ("fundamental", "Fundamental Analyst", ai_engine.run_fundamental_analysis),
                ("sentiment", "Sentiment Analyst", ai_engine.run_sentiment_analysis),
                ("risk", "Risk Analyst", ai_engine.run_risk_analysis),
            ]

            from concurrent.futures import ThreadPoolExecutor as _TPE
            results: dict = {}
            with _TPE(max_workers=4) as pool:
                futs = {pool.submit(fn, data): (key, label) for key, label, fn in stages}
                for fut in list(futs):
                    key, label = futs[fut]
                    try:
                        text = fut.result(timeout=180)
                    except Exception as e:  # noqa: BLE001
                        text = f"[error: {e}]"
                    results[key] = text
                    yield f"event: analyst\ndata: {json.dumps({'key': key, 'label': label, 'text': text})}\n\n"

            yield f"event: stage\ndata: {json.dumps({'stage': 'synthesis', 'message': 'Master Strategist synthesizing (extended thinking)...'})}\n\n"
            synth = ai_engine.run_master_synthesis(
                sym, results.get("technical", ""), results.get("fundamental", ""),
                results.get("sentiment", ""), results.get("risk", ""))
            yield f"event: synthesis\ndata: {json.dumps({'text': synth})}\n\n"
            yield f"event: done\ndata: {json.dumps({'symbol': sym, 'model': OPUS})}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return Response(gen(), headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


# ─────────────────────────────────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_state()
    if not ANTHROPIC_API_KEY:
        print("\n  ⚠  ANTHROPIC_API_KEY not set — AI features disabled.\n")
    live_str = "LIVE (Yahoo / Zacks / OpenInsider / Kalshi / News)" if USE_LIVE_DATA else "DEMO"
    print("\n" + "=" * 64)
    print("  AI TRADING RESEARCH TERMINAL")
    print(f"  Model:   {DEFAULT_MODEL}  (extended thinking + prompt caching)")
    print(f"  Mode:    {'PAPER' if PAPER_TRADING else 'LIVE'} TRADING")
    print(f"  Data:    {live_str}")
    print(f"  Capital: ${state['cash']:,.2f}  (initial ${INITIAL_CASH:,.2f})")
    print("  " + "-" * 60)
    print("  Sources: Yahoo, Zacks, OpenInsider, Kalshi,")
    print("           CNBC, MarketWatch, WSJ, Benzinga, Seeking Alpha,")
    print("           Reuters, Investing.com, CNN Fear & Greed")
    print("=" * 64)
    if DEMO_MODE:
        print("\n  Schwab API pending — ready for connection.")
    print(f"  Opening browser at http://127.0.0.1:{PORT}\n")
    threading.Thread(target=update_prices_loop, daemon=True).start()
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    socketio.run(app, debug=False, port=PORT, allow_unsafe_werkzeug=True)
