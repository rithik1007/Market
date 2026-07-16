"""
AI Analysis Module — Azure OpenAI powered market brief & stock reasoning.
Takes hard data from the screener and produces actionable AI insights.
"""

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
logger = logging.getLogger(__name__)

# Azure OpenAI client (lazy init)
_client = None


def _get_client():
    global _client
    if _client is None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        if not endpoint or not api_key:
            return None
        _client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return _client


def _get_deployment():
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-1-chat-2025-11-13")


def _get_temperature():
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0.4"))
    except ValueError:
        return 0.4


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> Optional[str]:
    """Call Azure OpenAI and return the text response."""
    client = _get_client()
    if client is None:
        logger.warning("Azure OpenAI not configured — skipping AI analysis")
        return None
    try:
        resp = client.chat.completions.create(
            model=_get_deployment(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=_get_temperature(),
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Azure OpenAI call failed: {e}")
        return None


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert NSE (National Stock Exchange of India) trader, portfolio strategist, and technical analyst.
You receive ONLY hard, verified data — price, volume, RSI, MACD, MAs, conviction scores, pattern detection, and trade levels.
You also receive the full universe of scanned stocks filtered to the trader's budget.

Your job is to produce ACTIONABLE TRADE RECOMMENDATIONS:
1. From the affordable_stocks list, identify the BEST stocks to buy right now or watch.
2. For BUY picks: specify exact entry price, stop loss, targets, qty, and ₹ amounts.
3. For WATCH picks: specify the trigger price and condition to enter.
4. Rank stocks by which one you'd put money on FIRST.
5. Give an overall portfolio plan: how many stocks to trade, capital to deploy vs keep in cash.
6. If market is bearish, recommend the best defensive picks or cheapest stocks with upside potential.

Rules:
- Never fabricate data. Only use numbers from the scan.
- Be brutally honest. If a stock looks weak, say so.
- The trader can ONLY buy stocks priced within their budget. Never recommend a stock they can't afford (price > budget).
- For small budgets (under ₹5,000): focus on 1-2 stocks max, allocate 50-80% to the best pick. Don't split tiny capital across many positions.
- qty = how many shares the trader can buy with allocated capital at entry price. Must be at least 1. If can't afford 1 share, skip that stock.
- All amounts in Indian Rupees (₹).
- IMPORTANT: Always recommend at least 1-3 stocks from the affordable list, even in bearish markets.
- TIMEFRAME: The user will specify a timeframe (Intraday, 1-2 Days, 3-5 Days, 1 Week, 2 Weeks, 1 Month). Tailor holding_period, targets, stop losses, and entry/exit timing to match this timeframe exactly.
Output valid JSON only — no markdown, no code fences."""


def generate_ai_analysis(scan_results: dict, capital: int = 100000, timeframe: str = "Intraday") -> Optional[dict]:
    """
    Generate AI-powered analysis with concrete trade plans.
    Filters the full stock universe by the user's budget.
    """
    # Build compact data payload for LLM (minimize tokens)
    top_picks_data = []
    for p in scan_results.get("top_picks", [])[:8]:
        top_picks_data.append({
            "ticker": p["ticker"],
            "sector": p["sector"],
            "close": p["close"],
            "change_pct": p["change_pct"],
            "pattern": p["pattern_name"],
            "entry": p["trade_levels"]["entry_zone"],
            "stop_loss": p["trade_levels"]["stop_loss"],
            "target_1": p["trade_levels"]["target_1"],
            "target_2": p["trade_levels"]["target_2"],
            "target_3": p["trade_levels"]["target_3"],
            "risk_reward": p["trade_levels"]["risk_reward"],
            "rsi": p["rsi"],
            "volume_ratio": p["volume_ratio"],
            "above_50dma": p["above_50dma"],
            "above_200dma": p["above_200dma"],
            "macd_bullish": p["macd_bullish"],
            "conviction_score": p.get("conviction_score", 0),
            "conviction_verdict": p.get("conviction_verdict", ""),
            "volume_confirmed": p.get("volume_confirmed", False),
            "fake_breakout": p.get("fake_breakout", {}).get("is_fake", False),
            "fake_confidence": p.get("fake_breakout", {}).get("confidence", 0),
            "institutional": p.get("institutional", {}).get("institutional", False),
            "inst_score": p.get("institutional", {}).get("score", 0),
            "confirmation_strength": p.get("confirmation", {}).get("strength", 0),
            "risk_factors": p.get("risk_factors", []),
        })

    # Filter full stock universe by budget — only stocks user can afford
    universe = scan_results.get("stock_universe", [])
    affordable = sorted(
        [s for s in universe if s["close"] <= capital],
        key=lambda s: (-s["rsi"] if s["rsi"] < 70 else 0, -s["volume_ratio"]),
    )
    # Send top 25 affordable stocks (sorted by technical strength) to keep tokens low
    affordable_for_ai = affordable[:25]

    index_data = scan_results.get("index_sentiment", [])
    oi_data = scan_results.get("oi_sentiment")
    sector_data = scan_results.get("sector_analysis", [])

    payload = {
        "capital": f"₹{capital:,}",
        "timeframe": timeframe,
        "total_scanned": scan_results["total_scanned"],
        "breakouts_found": scan_results["breakouts_found"],
        "affordable_stocks_count": len(affordable),
        "index_sentiment": index_data,
        "oi_sentiment": oi_data,
        "top_picks": top_picks_data,
        "affordable_stocks": affordable_for_ai,
        "sector_analysis": sector_data[:8],
        "avoid_count": len(scan_results.get("avoid_stocks", [])),
        "caution_alerts": scan_results.get("caution_alerts", []),
    }

    has_breakouts = scan_results["breakouts_found"] > 0

    if has_breakouts:
        user_prompt = _build_trade_prompt(payload, capital, timeframe)
    else:
        user_prompt = _build_no_breakout_prompt(payload, capital, timeframe)

    raw = _call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=3000)
    if raw is None:
        return None

    # Parse JSON from response
    return _parse_ai_response(raw)


def _build_trade_prompt(payload: dict, capital: int = 100000, timeframe: str = "Intraday") -> str:
    """Prompt when breakouts ARE found."""
    cap_fmt = f"₹{capital:,}"
    return f"""Analyze this NSE breakout scan. The trader has {cap_fmt} to invest.
TIMEFRAME: The trader wants recommendations for the **{timeframe}** timeframe.
Tailor ALL recommendations (entry timing, holding period, targets, stop losses) to this timeframe.
For shorter timeframes (Intraday, 1-2 Days), focus on momentum and quick moves.
For longer timeframes (1-2 Weeks, 1 Month), focus on trend strength and swing setups.

You have two data sources:
1. "top_picks" — stocks with confirmed breakout patterns (strongest signals)
2. "affordable_stocks" — ALL scanned stocks priced ≤ {cap_fmt}/share with their technicals (RSI, volume, trend, MAs)

ONLY recommend stocks the trader can afford (price ≤ {cap_fmt}). Calculate qty = floor({cap_fmt} * allocation% / price).

Return this exact JSON:
{{
  "market_brief": "2-4 sentence market assessment",
  "overall_bias": "BULLISH" or "BEARISH" or "NEUTRAL",
  "portfolio_plan": {{
    "stocks_to_trade": 2,
    "capital_to_deploy": "₹xx,xxx",
    "capital_in_cash": "₹xx,xxx",
    "deploy_pct": 60,
    "strategy_note": "1 sentence approach"
  }},
  "trade_plans": [
    {{
      "rank": 1,
      "ticker": "STOCK",
      "action": "BUY" or "WAIT" or "AVOID",
      "conviction": "HIGH" or "MEDIUM" or "LOW",
      "reasoning": "2-3 sentences grounded in data",
      "entry_price": "₹xxx",
      "stop_loss": "₹xxx",
      "target_1": "₹xxx",
      "target_2": "₹xxx",
      "holding_period": "Intraday / 1-3 Days / 1-2 Weeks / Swing",
      "entry_timing": "When exactly to enter — e.g. 'Buy on open if gaps above ₹xxx' or 'Wait for 15m candle close above ₹xxx' or 'Enter on pullback to ₹xxx support'",
      "exit_timing": "When exactly to exit — e.g. 'Book 50% at T1, trail SL to entry for rest' or 'Exit by 3:15 PM if intraday' or 'Exit if RSI crosses 70 on 15m chart'",
      "capital_to_invest": "₹xx,xxx",
      "capital_pct": 20,
      "qty": 10,
      "expected_profit_t1": "₹x,xxx",
      "expected_profit_t2": "₹x,xxx",
      "max_loss": "₹x,xxx",
      "risk_reward_note": "If SL hits lose ₹X, if T1 hits gain ₹Y"
    }}
  ],
  "risk_warning": "1 sentence key risk",
  "best_pick_summary": "Which stock would you bet on first and why"
}}

Scan Data:
{json.dumps(payload, default=str)}"""


def _build_no_breakout_prompt(payload: dict, capital: int = 100000, timeframe: str = "Intraday") -> str:
    """Prompt when NO breakouts found — bearish/flat market."""
    cap_fmt = f"₹{capital:,}"
    return f"""NSE scan found 0 breakouts today. But the trader has {cap_fmt} and wants to know WHAT to buy.
TIMEFRAME: The trader wants recommendations for the **{timeframe}** timeframe.
Tailor ALL recommendations (entry timing, holding period, targets, stop losses) to this timeframe.

You have "affordable_stocks" — ALL scanned stocks priced ≤ {cap_fmt}/share with their RSI, volume ratio, trend, and MA positions. Use this data to find the BEST stocks within budget.

Even in a bearish market, identify 3-5 stocks from affordable_stocks that:
- Have RSI between 30-50 (oversold bounce candidates)
- Or have strong volume ratio (>1.0) showing accumulation
- Or are above 50DMA/200DMA (relative strength)
- Or are in a strong sector (Pharma, FMCG, Telecom)

Return this JSON:
{{
  "market_brief": "3-5 sentences on WHY no breakouts — be specific about index levels.",
  "overall_bias": "BEARISH" or "NEUTRAL",
  "portfolio_plan": {{
    "stocks_to_trade": 0,
    "capital_to_deploy": "₹0",
    "capital_in_cash": "{cap_fmt}",
    "deploy_pct": 0,
    "strategy_note": "Clear instruction on what to do with {cap_fmt}"
  }},
  "trade_plans": [],
  "watchlist": [
    {{
      "ticker": "STOCK from affordable_stocks",
      "sector": "sector",
      "current_price": "₹xxx",
      "why_watch": "1-2 sentences — reference actual RSI, volume, trend data from the scan",
      "trigger_level": "₹xxx — exact price to enter",
      "entry_timing": "When to enter — e.g. 'Buy if 15m candle closes above trigger' or 'Enter on open if gaps up'",
      "exit_timing": "When to exit — e.g. 'Book at ₹xxx (+5%), SL at ₹xxx'",
      "qty_at_trigger": N,
      "investment_amount": "₹xxx"
    }}
  ],
  "re_entry_conditions": "2-3 sentences — specific conditions to deploy capital",
  "risk_warning": "1 sentence risk",
  "best_pick_summary": "Name the #1 affordable stock to watch and exact entry condition"
}}

Scan Data:
{json.dumps(payload, default=str)}"""


def _parse_ai_response(raw: str) -> Optional[dict]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    try:
        cleaned = raw.strip()
        # Strip markdown code fences
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        # Try direct JSON parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON object from mixed content
        import re
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            return json.loads(json_match.group())

        logger.error(f"AI response was not valid JSON: {raw[:200]}")
        return {"market_brief": raw, "trade_plans": [], "watchlist": [],
                "risk_warning": "", "overall_bias": "NEUTRAL",
                "portfolio_plan": None, "best_pick_summary": ""}
    except json.JSONDecodeError:
        logger.error(f"AI response was not valid JSON: {raw[:200]}")
        return {"market_brief": raw, "trade_plans": [], "watchlist": [],
                "risk_warning": "", "overall_bias": "NEUTRAL",
                "portfolio_plan": None, "best_pick_summary": ""}


def is_configured() -> bool:
    """Check if Azure OpenAI credentials are present."""
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))


# ──────────────────────────────────────────────
# Single Stock Analyzer — Intraday / Swing AI
# ──────────────────────────────────────────────

STOCK_ANALYZER_SYSTEM = """You are an expert NSE intraday and swing trader.
You receive hard technical data for a single stock — OHLCV, RSI, MACD, SMAs, Bollinger Bands, ATR, VWAP, support/resistance levels.
Your job is to give a PRECISE, ACTIONABLE intraday and short-term trading plan.

Rules:
- Be specific with price levels. No vague advice.
- Give EXACT entry price, stop loss, and 2-3 targets.
- Specify TIME-BASED rules: when to enter, when to exit, what to do at market open vs close.
- Include risk management: position size suggestion for ₹1,00,000 capital.
- If the stock looks bad for intraday, say so and suggest swing levels instead.
- All prices in ₹. Output valid JSON only — no markdown, no code fences."""


def analyze_single_stock(stock_data: dict) -> Optional[dict]:
    """Generate AI intraday/swing analysis for a single stock."""
    user_prompt = f"""Analyze this stock for INTRADAY and short-term trading:

{json.dumps(stock_data, default=str)}

Return this exact JSON:
{{
  "ticker": "{stock_data.get('ticker', '')}",
  "verdict": "STRONG BUY" or "BUY" or "NEUTRAL" or "AVOID" or "SHORT",
  "brief": "2-3 sentence assessment of current setup",
  "intraday_plan": {{
    "bias": "BULLISH" or "BEARISH" or "SIDEWAYS",
    "entry_price": "₹xxx",
    "entry_timing": "Exact condition — e.g. 'Buy above ₹xxx after 9:30 AM if 5m candle closes above VWAP'",
    "stop_loss": "₹xxx",
    "target_1": "₹xxx",
    "target_2": "₹xxx",
    "target_3": "₹xxx",
    "exit_timing": "When to book profits and when to cut — e.g. 'Book 50% at T1, trail SL to cost. Exit all by 3:15 PM'",
    "risk_reward": "x:1",
    "qty_for_1L": N,
    "max_loss": "₹xxx",
    "max_profit": "₹xxx"
  }},
  "swing_plan": {{
    "bias": "BULLISH" or "BEARISH" or "SIDEWAYS",
    "entry_price": "₹xxx",
    "entry_timing": "Condition to enter for 3-10 day hold",
    "stop_loss": "₹xxx",
    "target_1": "₹xxx",
    "target_2": "₹xxx",
    "exit_timing": "When to exit swing trade",
    "holding_period": "X days",
    "risk_reward": "x:1"
  }},
  "key_levels": {{
    "resistance_1": "₹xxx",
    "resistance_2": "₹xxx",
    "support_1": "₹xxx",
    "support_2": "₹xxx",
    "vwap": "₹xxx",
    "pivot": "₹xxx"
  }},
  "signals": [
    "Bullish/bearish signal 1",
    "Signal 2",
    "Signal 3"
  ],
  "avoid_if": "Condition when this trade is invalid — e.g. 'Avoid if opens below ₹xxx with gap down'"
}}"""

    raw = _call_llm(STOCK_ANALYZER_SYSTEM, user_prompt, max_tokens=2000)
    if raw is None:
        return None
    return _parse_ai_response(raw)
