"""
ai_engine.py — Deep AI Analysis & Prediction Engine
Powered by Claude Opus 4.7 with extended thinking + prompt caching.

Provides multi-pass analyst pipelines, streaming chat, structured predictions,
and self-paced tool-use research.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Iterator

import anthropic
import data_feeds

# ═══════════════════════════════════════════════════════════════════════
#  MODEL CONFIG
# ═══════════════════════════════════════════════════════════════════════

OPUS = "claude-opus-4-7"
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"

# Default model for the heaviest analyses
DEFAULT_MODEL = OPUS
# Model used for fast surveys / cheap calls
FAST_MODEL = HAIKU

THINKING_BUDGET_DEEP = 6000   # Master strategist – deep reasoning
THINKING_BUDGET_MID = 3000    # Per-analyst passes
THINKING_BUDGET_QUICK = 1500  # Quick analysis / chat

MAX_TOKENS_DEEP = 6000
MAX_TOKENS_STD = 3000
MAX_TOKENS_QUICK = 2000


# ═══════════════════════════════════════════════════════════════════════
#  ANALYST PROMPTS — these are CACHED to cut latency + cost on repeat calls
# ═══════════════════════════════════════════════════════════════════════

TECHNICAL_ANALYST = """You are a quantitative technical analyst with 30 years on a Wall Street trading desk.
Analyze ONLY the technical data provided. Deliver:

1. TREND: bullish / bearish / neutral with concrete evidence (price action, MA structure, momentum).
2. SUPPORT / RESISTANCE: 2-3 levels each, with reasoning grounded in swing highs/lows or MAs.
3. MOMENTUM: RSI interpretation, divergence detection, MACD-style crossover read if data permits.
4. MOVING AVERAGES: cross signals (10/20/50/200 if visible), distance from each MA in %.
5. VOLUME PROFILE: notable accumulation/distribution days, volume confirmation of trend.
6. PATTERN: identify any classical pattern (flag, wedge, H&S, double-top, base, breakout setup).
7. PRICE TARGETS: short-term and intermediate targets with method (measured move, Fib, MA target).
8. R/R: risk/reward at current entry with explicit stop distance.

Be precise with numbers. Cite the data you used. No fluff."""

FUNDAMENTAL_ANALYST = """You are a senior fundamental equity research analyst at a top-tier sell-side firm.
Analyze ONLY the fundamental data provided. Deliver:

1. VALUATION: overvalued / fairly valued / undervalued vs sector and history. Cite multiples.
2. KEY RATIOS: P/E, P/S, EV/EBITDA, gross/op/net margin, ROIC – flag the standouts.
3. EARNINGS QUALITY: revenue mix, margin trajectory, FCF conversion, accruals red flags.
4. GROWTH: top-line, bottom-line, segment growth – are estimates beatable?
5. COMPETITIVE MOAT: durable advantages, pricing power, threats.
6. CAPITAL ALLOCATION: buybacks, dividends, M&A, capex discipline.
7. FAIR VALUE: provide an explicit fair-value estimate with the methodology used.
8. CATALYSTS: upcoming earnings dates, product cycles, macro sensitivities.

Be data-driven. If a critical data point is missing, say so explicitly."""

SENTIMENT_ANALYST = """You are a market sentiment and behavioral finance specialist.
Analyze ONLY the news, insider activity, fear/greed, and prediction-market data.

1. NARRATIVE: in 1-2 sentences, what is the story the tape is telling about this name right now?
2. SENTIMENT SCORE: integer -10 (max bearish) to +10 (max bullish) with explicit weighting.
3. NEWS CLUSTER: identify the dominant theme (earnings, regulation, product, macro, takeover...).
4. INSIDER SIGNAL: cluster buys vs sells, who is trading (CEO/CFO vs lower-level), $ size.
5. POSITIONING: short interest, options skew, prediction-market odds (where available).
6. CROWD vs CONTRARIAN: is sentiment so extreme it is a fade signal?
7. CATALYSTS / TRAPS: incoming events that could whipsaw the narrative.

Be specific about what data SAYS vs what it IMPLIES."""

RISK_ANALYST = """You are a portfolio risk manager. Build the bear case and quantify the downside.

1. WORST CASE: realistic 3-month drawdown scenario with probability and trigger.
2. KEY RISKS: rank top 5 (idiosyncratic, sector, macro, regulatory, technical).
3. CORRELATION: beta to SPY/QQQ, dollar exposure, sector clustering risk.
4. VOLATILITY: expected 1-month range using ATR/IV; gap risk around earnings.
5. OPTIONS-IMPLIED MOVE: if data given, decode the implied move into a concrete $ range.
6. TAIL RISKS: black-swan paths (fraud, dilution, regulatory action, supply shock).
7. POSITION SIZING: max position for a $30,000 account using 1-2% portfolio risk.
8. HEDGES: practical hedges (puts, pair trade, sector short) with cost.

Be honest. The point is to find what kills the trade."""

MASTER_STRATEGIST = """You are the Chief Investment Strategist. You receive 4 specialist reports
(Technical, Fundamental, Sentiment, Risk) plus the raw market context. Synthesize everything
into ONE crisp, actionable recommendation.

Use this exact structure (markdown):

## VERDICT
**ACTION:** BUY / SELL / HOLD / TRIM / ADD
**CONVICTION:** X/10
**TIME HORIZON:** intraday / swing (days) / position (weeks-months)

One sharp sentence: what to do and why right now.

## PRICE TARGETS
| Horizon | Target | Probability |
|---|---|---|
| 1 week | $X | XX% |
| 1 month | $X | XX% |
| 3 months | $X | XX% |
| Stop loss | $X | — |
| Max position | XX shares ($X) | — |

## CORE THESIS
3-4 sentences. The single most important reason this trade works (or doesn't).

## CATALYSTS
- Upcoming events that could move the stock (earnings, product, regulatory, macro). Date them.

## TOP RISKS
- Top 3 with probability + impact rating.

## TRADE PLAN
- Entry zone: $X – $X
- Initial stop: $X (— X% from entry)
- First scale-out: $X (— Y% gain)
- Runner target: $X
- Position management rules.

## CONFIDENCE NOTES
Where the analysts agreed, where they diverged, and how you weighted them.

Frame as research, not licensed financial advice. This is a paper-trading account."""

CHAT_SYSTEM = """You are an elite trading research assistant powered by Claude Opus 4.7 with extended reasoning.
You support a trader running a $30,000 paper-trading account on a multi-source research terminal.

LIVE DATA YOU HAVE ACCESS TO (via the user's terminal, surfaced to you in [MARKET CONTEXT] blocks):
- Yahoo Finance: real-time quotes, OHLCV history, fundamentals
- Zacks: ranks, price targets, key statistics
- OpenInsider: insider buys/sells with size and role
- Kalshi: prediction-market odds
- News: CNBC, MarketWatch, WSJ, Benzinga, Seeking Alpha, Yahoo Finance, Reuters
- CNN Fear & Greed Index
- Computed signals: RSI, SMA10/20, ATR, derived buy/sell/hold

HOW TO ANSWER:
- Be direct. Lead with the conclusion, follow with the supporting evidence.
- Cite the specific data point you used (e.g. "RSI is 72 — overbought").
- For trade ideas, ALWAYS include: entry zone, stop loss, target(s), position size, time horizon.
- Cross-reference multiple sources. A single news headline is not a thesis.
- Discuss option Greeks plainly. Translate IV % into expected $ moves.
- When sentiment and technicals diverge, name the divergence and pick a side.
- Flag uncertainty explicitly. Do not invent numbers.
- This is research, not licensed financial advice. Paper trading — no real capital at risk."""


# ═══════════════════════════════════════════════════════════════════════
#  RETRY HELPER
# ═══════════════════════════════════════════════════════════════════════

def _with_retry(fn: Callable, *args, retries: int = 2, backoff: float = 1.5, **kwargs):
    """Call fn with exponential backoff. Surfaces the last exception text."""
    last = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except anthropic.APIStatusError as e:
            last = e
            if e.status_code in (400, 401, 403, 404):
                # Non-retryable
                raise
            time.sleep(backoff * (2 ** attempt))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(backoff * (2 ** attempt))
    raise last  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
#  ENGINE
# ═══════════════════════════════════════════════════════════════════════

class AIAnalysisEngine:
    """Multi-pass analysis engine using Claude Opus 4.7."""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.analysis_cache: dict = {}
        self._executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="ai-engine")

    # ── low-level API call ────────────────────────────────────────────
    def _call_claude(
        self,
        system_prompt: str,
        user_content: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = MAX_TOKENS_STD,
        thinking_budget: int | None = None,
        cache_system: bool = True,
    ) -> str:
        """Single Claude call with prompt caching + optional extended thinking."""
        system_param: list | str
        if cache_system:
            system_param = [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            system_param = system_prompt

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_param,
            "messages": [{"role": "user", "content": user_content}],
        }
        if thinking_budget and thinking_budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            # Anthropic requires temperature=1 with thinking
            kwargs["temperature"] = 1.0
            # max_tokens must exceed budget for thinking calls
            if max_tokens <= thinking_budget:
                kwargs["max_tokens"] = thinking_budget + 2048

        try:
            resp = _with_retry(self.client.messages.create, **kwargs)
        except Exception as e:  # noqa: BLE001
            return f"[Analysis Error: {e}]"

        # When thinking is on the response has thinking + text blocks
        text_chunks = []
        for block in resp.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_chunks.append(block.text)
        return "\n".join(text_chunks).strip() or "[Empty response]"

    def stream_claude(
        self,
        system_prompt: str,
        user_content: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = MAX_TOKENS_STD,
        thinking_budget: int | None = None,
        history: list | None = None,
    ) -> Iterator[str]:
        """Yield text deltas as Claude streams the response."""
        messages = list(history or [])
        messages.append({"role": "user", "content": user_content})

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": messages,
        }
        if thinking_budget and thinking_budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            kwargs["temperature"] = 1.0
            if max_tokens <= thinking_budget:
                kwargs["max_tokens"] = thinking_budget + 2048

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "type", "") == "text_delta":
                            yield delta.text
        except Exception as e:  # noqa: BLE001
            yield f"\n\n[stream error: {e}]"

    # ── data gathering ────────────────────────────────────────────────
    def gather_data(self, symbol: str) -> dict:
        """Pull every source in parallel; tolerate individual failures."""
        results: dict = {"symbol": symbol.upper(), "timestamp": datetime.now().isoformat()}
        tasks = [
            ("quote", data_feeds.yahoo_quote, (symbol,)),
            ("history_3mo", data_feeds.yahoo_history, (symbol, "3mo", "1d")),
            ("history_1y", data_feeds.yahoo_history, (symbol, "1y", "1wk")),
            ("fundamentals", data_feeds.yahoo_fundamentals, (symbol,)),
            ("zacks", data_feeds.zacks_rating, (symbol,)),
            ("insider_trades", data_feeds.openinsider_latest, (symbol,)),
            ("news", data_feeds.news_for_symbol, (symbol,)),
            ("fear_greed", data_feeds.fear_greed_index, ()),
        ]
        futures = {self._executor.submit(fn, *args): key for key, fn, args in tasks}
        for fut in as_completed(futures, timeout=20):
            key = futures[fut]
            try:
                results[key] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[key] = {"error": str(e)}
        return results

    # ── individual analyst passes ─────────────────────────────────────
    def run_technical_analysis(self, data: dict) -> str:
        content = json.dumps({
            "symbol": data.get("symbol"),
            "quote": data.get("quote", {}),
            "history_3mo_daily": (data.get("history_3mo") or [])[-45:],
            "history_1y_weekly": (data.get("history_1y") or [])[-30:],
        }, indent=2, default=str)
        return self._call_claude(
            TECHNICAL_ANALYST,
            f"Technical readout for {data['symbol']}:\n{content}",
            thinking_budget=THINKING_BUDGET_MID,
            max_tokens=MAX_TOKENS_STD,
        )

    def run_fundamental_analysis(self, data: dict) -> str:
        content = json.dumps({
            "symbol": data.get("symbol"),
            "quote": data.get("quote", {}),
            "fundamentals": data.get("fundamentals", {}),
            "zacks": data.get("zacks", {}),
        }, indent=2, default=str)
        return self._call_claude(
            FUNDAMENTAL_ANALYST,
            f"Fundamental analysis for {data['symbol']}:\n{content}",
            thinking_budget=THINKING_BUDGET_MID,
            max_tokens=MAX_TOKENS_STD,
        )

    def run_sentiment_analysis(self, data: dict) -> str:
        content = json.dumps({
            "symbol": data.get("symbol"),
            "news": (data.get("news") or [])[:12],
            "insider_trades": (data.get("insider_trades") or [])[:15],
            "fear_greed": data.get("fear_greed", {}),
        }, indent=2, default=str)
        return self._call_claude(
            SENTIMENT_ANALYST,
            f"Sentiment readout for {data['symbol']}:\n{content}",
            thinking_budget=THINKING_BUDGET_MID,
            max_tokens=MAX_TOKENS_STD,
        )

    def run_risk_analysis(self, data: dict) -> str:
        content = json.dumps({
            "symbol": data.get("symbol"),
            "quote": data.get("quote", {}),
            "history_3mo_daily": (data.get("history_3mo") or [])[-45:],
            "fundamentals": data.get("fundamentals", {}),
            "fear_greed": data.get("fear_greed", {}),
        }, indent=2, default=str)
        return self._call_claude(
            RISK_ANALYST,
            f"Risk profile for {data['symbol']}:\n{content}",
            thinking_budget=THINKING_BUDGET_MID,
            max_tokens=MAX_TOKENS_STD,
        )

    def run_master_synthesis(self, symbol: str, technical: str, fundamental: str,
                             sentiment: str, risk: str) -> str:
        content = (
            f"SYMBOL: {symbol}\n\n"
            "=== TECHNICAL ANALYST REPORT ===\n" + technical +
            "\n\n=== FUNDAMENTAL ANALYST REPORT ===\n" + fundamental +
            "\n\n=== SENTIMENT ANALYST REPORT ===\n" + sentiment +
            "\n\n=== RISK ANALYST REPORT ===\n" + risk +
            "\n\nProduce the final actionable synthesis using the required structure."
        )
        return self._call_claude(
            MASTER_STRATEGIST,
            content,
            model=OPUS,
            thinking_budget=THINKING_BUDGET_DEEP,
            max_tokens=MAX_TOKENS_DEEP,
        )

    # ── full 5-pass pipeline ──────────────────────────────────────────
    def full_analysis(self, symbol: str, progress_callback: Callable | None = None) -> dict:
        result = {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat(),
            "model": OPUS,
            "thinking_enabled": True,
            "status": "running",
        }
        try:
            if progress_callback:
                progress_callback("Gathering live data from all sources...")
            data = self.gather_data(symbol)
            result["data_sources"] = [k for k in data if k not in ("symbol", "timestamp")]

            if progress_callback:
                progress_callback("Running 4 specialist analysts in parallel...")

            futs = {
                "technical":  self._executor.submit(self.run_technical_analysis, data),
                "fundamental":self._executor.submit(self.run_fundamental_analysis, data),
                "sentiment":  self._executor.submit(self.run_sentiment_analysis, data),
                "risk":       self._executor.submit(self.run_risk_analysis, data),
            }
            analyses = {name: f.result(timeout=180) for name, f in futs.items()}

            result["technical_analysis"]   = analyses["technical"]
            result["fundamental_analysis"] = analyses["fundamental"]
            result["sentiment_analysis"]   = analyses["sentiment"]
            result["risk_analysis"]        = analyses["risk"]

            if progress_callback:
                progress_callback("Master Strategist synthesizing all reports (extended thinking)...")

            result["master_synthesis"] = self.run_master_synthesis(
                symbol,
                analyses["technical"], analyses["fundamental"],
                analyses["sentiment"], analyses["risk"],
            )

            result["status"] = "complete"
            result["api_calls"] = 5
            self.analysis_cache[symbol.upper()] = result
        except Exception as e:  # noqa: BLE001
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
        return result

    # ── faster modes ──────────────────────────────────────────────────
    def quick_analysis(self, symbol: str) -> str:
        data = self.gather_data(symbol)
        content = json.dumps({
            "symbol": data.get("symbol"),
            "quote": data.get("quote", {}),
            "history_recent": (data.get("history_3mo") or [])[-20:],
            "fundamentals": data.get("fundamentals", {}),
            "zacks": data.get("zacks", {}),
            "insider_trades": (data.get("insider_trades") or [])[:8],
            "news": [{"title": n.get("title"), "source": n.get("source")}
                     for n in (data.get("news") or [])[:10]],
            "fear_greed": data.get("fear_greed", {}),
        }, indent=2, default=str)
        return self._call_claude(
            MASTER_STRATEGIST,
            f"Quick but rigorous analysis of {symbol} using ALL of this data:\n{content}",
            thinking_budget=THINKING_BUDGET_QUICK,
            max_tokens=MAX_TOKENS_STD,
        )

    def compare_stocks(self, symbols: list) -> str:
        if not symbols:
            return "No symbols provided."
        all_data: dict = {}
        futs = {self._executor.submit(self.gather_data, s.upper()): s.upper()
                for s in symbols[:6]}
        for fut in as_completed(futs, timeout=45):
            sym = futs[fut]
            try:
                all_data[sym] = fut.result()
            except Exception as e:  # noqa: BLE001
                all_data[sym] = {"error": str(e)}

        compact = {}
        for sym, d in all_data.items():
            insiders = d.get("insider_trades") or []
            compact[sym] = {
                "quote": d.get("quote", {}),
                "zacks": d.get("zacks", {}),
                "fundamentals_keys": list((d.get("fundamentals", {}) or {}).get("fundamentals", {}).keys())[:8],
                "insider_summary": {
                    "buys":  sum(1 for t in insiders if "Purchase" in (t.get("trade_type", "") if isinstance(t, dict) else "")),
                    "sells": sum(1 for t in insiders if "Sale" in (t.get("trade_type", "") if isinstance(t, dict) else "")),
                },
                "recent_headlines": [n.get("title") for n in (d.get("news") or [])[:4] if isinstance(n, dict)],
            }

        return self._call_claude(
            """You are a portfolio strategist comparing stocks for a $30,000 paper-trading account.
Rank the candidates from best to worst. For each, state:
- Score 1-10
- Best entry zone + stop
- Position size in shares for this account
- Single sentence reason
End with a single allocation table.""",
            f"Compare these tickers using the bundled research:\n{json.dumps(compact, indent=2, default=str)}",
            thinking_budget=THINKING_BUDGET_QUICK,
            max_tokens=MAX_TOKENS_STD,
        )

    def market_brief(self) -> str:
        data: dict = {}
        tasks = [
            ("fear_greed",   data_feeds.fear_greed_index, ()),
            ("news",         data_feeds.aggregate_news, (18,)),
            ("trending",     data_feeds.yahoo_trending, ()),
            ("insider_buys", data_feeds.openinsider_top_buys, ()),
            ("market",       data_feeds.market_overview, ()),
        ]
        futures = {self._executor.submit(fn, *args): key for key, fn, args in tasks}
        for fut in as_completed(futures, timeout=20):
            key = futures[fut]
            try:
                data[key] = fut.result()
            except Exception as e:  # noqa: BLE001
                data[key] = {"error": str(e)}

        compact = {
            "fear_greed": data.get("fear_greed", {}),
            "indices": [{"name": q.get("name"), "price": q.get("price"),
                         "change_pct": q.get("change_pct")}
                        for q in (data.get("market", {}) or {}).get("indices", [])],
            "futures": [{"name": q.get("name"), "change_pct": q.get("change_pct")}
                        for q in (data.get("market", {}) or {}).get("futures", [])],
            "crypto":  [{"name": q.get("name"), "change_pct": q.get("change_pct")}
                        for q in (data.get("market", {}) or {}).get("crypto", [])],
            "top_headlines": [{"title": n.get("title"), "source": n.get("source")}
                              for n in (data.get("news") or [])[:14]
                              if isinstance(n, dict)],
            "trending": data.get("trending", []),
            "notable_insider_buys": [
                {"ticker": t.get("ticker"), "company": t.get("company"),
                 "insider": t.get("insider_name"), "value": t.get("value")}
                for t in (data.get("insider_buys") or [])[:10] if isinstance(t, dict)
            ],
        }

        return self._call_claude(
            """You are the morning desk strategist. Deliver a tight, structured market brief.

# MARKET MOOD
One paragraph. Use Fear/Greed + index futures + news tone.

# 5 STORIES THAT MATTER TODAY
Numbered list. One sentence each. Why it moves money.

# WATCHLIST
Bullets. Symbols pulled from trending + insider activity. For each: reason + setup.

# 3 TRADE IDEAS
For each: ticker, direction, entry zone, stop, target, time horizon, conviction (1-10).

# RISK WATCH
What kills the day's playbook. Two bullets.

Be punchy. Numbers, not adjectives.""",
            f"Today's data ({datetime.now().strftime('%Y-%m-%d %H:%M')}):\n{json.dumps(compact, indent=2, default=str)}",
            thinking_budget=THINKING_BUDGET_QUICK,
            max_tokens=MAX_TOKENS_STD,
        )
