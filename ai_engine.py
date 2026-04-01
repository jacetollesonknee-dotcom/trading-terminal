"""
ai_engine.py – Deep AI Analysis & Prediction Engine
Uses Claude to run multi-layered analysis across all data sources.
Provides structured predictions with confidence scoring.
"""

import json, threading, time, traceback
from datetime import datetime
import anthropic
import data_feeds

# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS PROMPTS  –  Specialized prompts for each analysis type
# ═══════════════════════════════════════════════════════════════════════

TECHNICAL_ANALYST = """You are a quantitative technical analyst with 30 years experience.
Analyze ONLY the technical data provided. Give:
- Trend direction (bullish/bearish/neutral) with reasoning
- Key support and resistance levels
- RSI interpretation and divergence signals
- Moving average crossover analysis
- Volume profile analysis
- Specific price targets based on technical patterns
- Risk/reward ratio for a trade entry at current price
Be precise with numbers. No fluff."""

FUNDAMENTAL_ANALYST = """You are a fundamental equity research analyst at a top Wall Street firm.
Analyze ONLY the fundamental data provided. Give:
- Valuation assessment (overvalued/fairly valued/undervalued)
- Key financial ratios and what they tell us
- Earnings quality assessment
- Growth trajectory analysis
- Competitive positioning
- Fair value estimate with methodology
Be data-driven and specific."""

SENTIMENT_ANALYST = """You are a market sentiment and behavioral finance specialist.
Analyze ONLY the news, insider activity, and sentiment data provided. Give:
- Overall sentiment score (-10 to +10)
- News narrative analysis (what's the story the market is telling?)
- Insider activity interpretation (are insiders bullish or bearish?)
- Prediction market signals (if available)
- Fear & Greed impact on this stock
- Contrarian signals (is sentiment too extreme?)
Be specific about what the data says vs. what it implies."""

RISK_ANALYST = """You are a risk management specialist and portfolio strategist.
Based on ALL data provided, assess:
- Maximum downside risk (worst case scenario with probability)
- Key risk factors (macro, sector, company-specific)
- Correlation risk with major indices
- Volatility assessment and expected range
- Options-implied expectations (if available)
- Black swan risk factors
- Position sizing recommendation for a $30,000 portfolio
Be conservative and thorough."""

MASTER_STRATEGIST = """You are the Chief Investment Strategist synthesizing all analyst reports.
You have reports from: Technical Analyst, Fundamental Analyst, Sentiment Analyst, Risk Analyst.

Synthesize ALL reports into a final recommendation with:

## VERDICT
One clear sentence: BUY, SELL, or HOLD with conviction level (1-10)

## PRICE TARGETS
- 1-Week Target: $X (probability: X%)
- 1-Month Target: $X (probability: X%)
- 3-Month Target: $X (probability: X%)
- Stop Loss: $X
- Maximum Position Size: X shares ($X value)

## THESIS
3-4 sentences explaining the core thesis

## KEY CATALYSTS
- Upcoming events that could move the stock (earnings, FDA, macro)

## RISK FACTORS
- Top 3 risks with probability and impact ratings

## TRADE PLAN
Specific entry, exit, and risk management instructions

Remember: Frame as research, not financial advice. This is paper trading."""


# ═══════════════════════════════════════════════════════════════════════
#  MULTI-PASS ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════

class AIAnalysisEngine:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.analysis_cache = {}

    def _call_claude(self, system_prompt: str, user_content: str,
                     model: str = "claude-sonnet-4-5-20250514", max_tokens: int = 1500) -> str:
        """Make a single Claude API call."""
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            return response.content[0].text
        except Exception as e:
            return f"[Analysis Error: {str(e)}]"

    def gather_data(self, symbol: str) -> dict:
        """Gather all available data for a symbol from all sources."""
        results = {}
        threads = []

        def run(key, fn, *args):
            try:
                results[key] = fn(*args)
            except Exception as e:
                results[key] = {"error": str(e)}

        tasks = [
            ("quote", data_feeds.yahoo_quote, symbol),
            ("history_3mo", data_feeds.yahoo_history, symbol, "3mo", "1d"),
            ("history_1y", data_feeds.yahoo_history, symbol, "1y", "1wk"),
            ("fundamentals", data_feeds.yahoo_fundamentals, symbol),
            ("zacks", data_feeds.zacks_rating, symbol),
            ("insider_trades", data_feeds.openinsider_latest, symbol),
            ("news", data_feeds.news_for_symbol, symbol),
            ("fear_greed", data_feeds.fear_greed_index),
        ]

        for task in tasks:
            key, fn = task[0], task[1]
            args = task[2:] if len(task) > 2 else ()
            t = threading.Thread(target=run, args=(key, fn, *args))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15)

        results["symbol"] = symbol.upper()
        results["timestamp"] = datetime.now().isoformat()
        return results

    def run_technical_analysis(self, data: dict) -> str:
        """Pass 1: Technical analysis."""
        content = json.dumps({
            "symbol": data.get("symbol"),
            "quote": data.get("quote", {}),
            "history_3mo": data.get("history_3mo", [])[-30:],  # last 30 days
            "history_1y_weekly": data.get("history_1y", [])[-26:],  # last 6 months weekly
        }, indent=2, default=str)
        return self._call_claude(TECHNICAL_ANALYST, f"Analyze {data['symbol']}:\n{content}")

    def run_fundamental_analysis(self, data: dict) -> str:
        """Pass 2: Fundamental analysis."""
        content = json.dumps({
            "symbol": data.get("symbol"),
            "quote": data.get("quote", {}),
            "fundamentals": data.get("fundamentals", {}),
            "zacks": data.get("zacks", {}),
        }, indent=2, default=str)
        return self._call_claude(FUNDAMENTAL_ANALYST, f"Analyze {data['symbol']}:\n{content}")

    def run_sentiment_analysis(self, data: dict) -> str:
        """Pass 3: Sentiment analysis."""
        content = json.dumps({
            "symbol": data.get("symbol"),
            "news": data.get("news", [])[:10],
            "insider_trades": data.get("insider_trades", [])[:10],
            "fear_greed": data.get("fear_greed", {}),
        }, indent=2, default=str)
        return self._call_claude(SENTIMENT_ANALYST, f"Analyze sentiment for {data['symbol']}:\n{content}")

    def run_risk_analysis(self, data: dict) -> str:
        """Pass 4: Risk analysis."""
        content = json.dumps({
            "symbol": data.get("symbol"),
            "quote": data.get("quote", {}),
            "history_3mo": data.get("history_3mo", [])[-30:],
            "fundamentals": data.get("fundamentals", {}),
            "fear_greed": data.get("fear_greed", {}),
        }, indent=2, default=str)
        return self._call_claude(RISK_ANALYST, f"Risk analysis for {data['symbol']}:\n{content}")

    def run_master_synthesis(self, symbol: str, technical: str, fundamental: str,
                             sentiment: str, risk: str) -> str:
        """Pass 5: Master strategist synthesizes all reports."""
        content = f"""SYMBOL: {symbol}

=== TECHNICAL ANALYST REPORT ===
{technical}

=== FUNDAMENTAL ANALYST REPORT ===
{fundamental}

=== SENTIMENT ANALYST REPORT ===
{sentiment}

=== RISK ANALYST REPORT ===
{risk}

Synthesize all reports into a final actionable recommendation."""

        return self._call_claude(
            MASTER_STRATEGIST, content,
            model="claude-sonnet-4-5-20250514",
            max_tokens=2500
        )

    def full_analysis(self, symbol: str, progress_callback=None) -> dict:
        """Run the complete 5-pass analysis pipeline."""
        result = {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat(),
            "status": "running"
        }

        try:
            # Step 1: Gather all data
            if progress_callback:
                progress_callback("Gathering live data from all sources...")
            data = self.gather_data(symbol)
            result["data_sources"] = list(data.keys())

            # Step 2: Run all 4 analyst passes in parallel
            if progress_callback:
                progress_callback("Running 4 AI analyst passes in parallel...")

            analyses = {}
            threads = []

            def run_analysis(name, fn, *args):
                analyses[name] = fn(*args)

            analysis_tasks = [
                ("technical", self.run_technical_analysis, data),
                ("fundamental", self.run_fundamental_analysis, data),
                ("sentiment", self.run_sentiment_analysis, data),
                ("risk", self.run_risk_analysis, data),
            ]

            for name, fn, *args in analysis_tasks:
                t = threading.Thread(target=run_analysis, args=(name, fn, *args))
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=60)

            result["technical_analysis"] = analyses.get("technical", "Error")
            result["fundamental_analysis"] = analyses.get("fundamental", "Error")
            result["sentiment_analysis"] = analyses.get("sentiment", "Error")
            result["risk_analysis"] = analyses.get("risk", "Error")

            # Step 3: Master synthesis
            if progress_callback:
                progress_callback("Master strategist synthesizing all reports...")

            result["master_synthesis"] = self.run_master_synthesis(
                symbol,
                analyses.get("technical", ""),
                analyses.get("fundamental", ""),
                analyses.get("sentiment", ""),
                analyses.get("risk", "")
            )

            result["status"] = "complete"
            result["api_calls"] = 5

            # Cache it
            self.analysis_cache[symbol.upper()] = result

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()

        return result

    def quick_analysis(self, symbol: str) -> str:
        """Faster single-pass analysis using all data at once."""
        data = self.gather_data(symbol)
        content = json.dumps({
            "symbol": data.get("symbol"),
            "quote": data.get("quote", {}),
            "history_recent": data.get("history_3mo", [])[-15:],
            "fundamentals": data.get("fundamentals", {}),
            "zacks": data.get("zacks", {}),
            "insider_trades": data.get("insider_trades", [])[:5],
            "news": [{"title": n.get("title"), "source": n.get("source")} for n in data.get("news", [])[:8]],
            "fear_greed": data.get("fear_greed", {}),
        }, indent=2, default=str)

        return self._call_claude(
            MASTER_STRATEGIST,
            f"Quick but thorough analysis of {symbol} using all this data:\n{content}",
            max_tokens=2000
        )

    def compare_stocks(self, symbols: list) -> str:
        """Compare multiple stocks and recommend the best one."""
        all_data = {}
        for sym in symbols[:5]:  # max 5
            all_data[sym] = self.gather_data(sym)

        content = json.dumps({sym: {
            "quote": d.get("quote", {}),
            "zacks": d.get("zacks", {}),
            "insider_summary": f"{len([t for t in d.get('insider_trades', []) if 'Purchase' in (t.get('trade_type',''))])} buys, "
                               f"{len([t for t in d.get('insider_trades', []) if 'Sale' in (t.get('trade_type',''))])} sells",
        } for sym, d in all_data.items()}, indent=2, default=str)

        return self._call_claude(
            """You are a portfolio strategist comparing stocks for a $30,000 paper trading account.
Compare the stocks and:
1. Rank them from best to worst opportunity
2. Explain your reasoning
3. Give specific allocation recommendations
4. Include entry prices and stop losses for each""",
            f"Compare these stocks:\n{content}",
            max_tokens=2000
        )

    def market_brief(self) -> str:
        """Generate a comprehensive market brief."""
        data = {}
        threads = []

        def run(key, fn, *args):
            try:
                data[key] = fn(*args)
            except:
                data[key] = {}

        tasks = [
            ("fear_greed", data_feeds.fear_greed_index),
            ("news", data_feeds.aggregate_news, 15),
            ("trending", data_feeds.yahoo_trending),
            ("insider_buys", data_feeds.openinsider_top_buys),
        ]

        for task in tasks:
            key, fn = task[0], task[1]
            args = task[2:] if len(task) > 2 else ()
            t = threading.Thread(target=run, args=(key, fn, *args))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15)

        content = json.dumps({
            "fear_greed": data.get("fear_greed", {}),
            "top_headlines": [{"title": n.get("title"), "source": n.get("source")}
                             for n in data.get("news", [])[:12] if isinstance(n, dict)],
            "trending_tickers": data.get("trending", []),
            "notable_insider_buys": data.get("insider_buys", [])[:8],
        }, indent=2, default=str)

        return self._call_claude(
            """You are a morning market strategist delivering the daily brief to a trading desk.
Synthesize all the data into a clear, actionable morning brief:
1. MARKET MOOD: Overall assessment based on Fear & Greed and news
2. KEY STORIES: Top 3-5 stories that will move markets today
3. STOCKS TO WATCH: Based on trending tickers and insider activity
4. TRADE IDEAS: 2-3 specific trade ideas based on the data
5. RISK WATCH: What could go wrong today
Keep it punchy and actionable.""",
            f"Today's data:\n{content}",
            max_tokens=2000
        )
