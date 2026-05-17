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
        return float(os.getenv("LLM_TEMPERATURE", "0.7"))
    except ValueError:
        return 0.7


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

SYSTEM_PROMPT = """You are an expert NSE (National Stock Exchange of India) breakout trader, portfolio strategist, and technical analyst.
You receive ONLY hard, verified data — price, volume, RSI, MACD, MAs, conviction scores, pattern detection, and trade levels.

Your job is to produce ACTIONABLE TRADE PLANS — tell the trader EXACTLY what to do:
1. For each stock: BUY / WAIT / AVOID with a clear reason.
2. For BUY picks: specify how much capital to allocate (as % of a ₹1,00,000 portfolio), the exact entry price, stop loss, and targets.
3. Calculate the expected profit in ₹ if Target 1 hits, and the max loss in ₹ if stop loss hits.
4. Suggest a holding period (Intraday / 1-3 days / 1-2 weeks / Swing 2-4 weeks).
5. Rank stocks by which one you'd put money on FIRST.
6. Give an overall portfolio plan: how many stocks to trade today, total capital to deploy vs keep in cash.
7. If NO breakouts are found, still give value: analyze why the market is weak, which sectors to watch, when to re-enter, and what price levels to watch on Nifty/BankNifty. Suggest stocks from the sector data that could break out next.

Rules:
- Never fabricate data. Only use numbers from the scan.
- Be brutally honest. If a breakout looks weak, say so.
- If market is bearish with 0 breakouts, recommend staying in cash and explain what conditions to wait for.
- Position size based on conviction: High conviction = 15-25% of capital, Medium = 8-12%, Low = skip.
- Flag contradictions (breakout + weak volume = suspicious, high conviction + fake breakout risk = reduce size).
- All amounts in Indian Rupees (₹).
Output valid JSON only — no markdown, no code fences."""


def generate_ai_analysis(scan_results: dict, capital: int = 100000) -> Optional[dict]:
    """
    Generate AI-powered analysis with concrete trade plans.
    Returns market brief + per-stock trade plans with amounts.
    Handles zero-breakout scenarios with watchlist advice.
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

    index_data = scan_results.get("index_sentiment", [])
    oi_data = scan_results.get("oi_sentiment")
    sector_data = scan_results.get("sector_analysis", [])

    payload = {
        "total_scanned": scan_results["total_scanned"],
        "breakouts_found": scan_results["breakouts_found"],
        "index_sentiment": index_data,
        "oi_sentiment": oi_data,
        "top_picks": top_picks_data,
        "sector_analysis": sector_data[:8],
        "avoid_count": len(scan_results.get("avoid_stocks", [])),
        "caution_alerts": scan_results.get("caution_alerts", []),
    }

    has_breakouts = scan_results["breakouts_found"] > 0

    if has_breakouts:
        user_prompt = _build_trade_prompt(payload, capital)
    else:
        user_prompt = _build_no_breakout_prompt(payload, capital)

    raw = _call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=3000)
    if raw is None:
        return None

    # Parse JSON from response
    return _parse_ai_response(raw)


def _build_trade_prompt(payload: dict, capital: int = 100000) -> str:
    """Prompt when breakouts ARE found."""
    cap_fmt = f"₹{capital:,}"
    return f"""Analyze this NSE breakout scan and return a JSON trade plan. The trader's available capital is {cap_fmt}. Only recommend stocks whose per-share price fits within this budget (trader must be able to buy at least 1 share). Calculate qty, amounts, and percentages relative to {cap_fmt}.

Return this exact JSON structure:
{{
  "market_brief": "2-4 sentence assessment of today's market",
  "overall_bias": "BULLISH" or "BEARISH" or "NEUTRAL",
  "portfolio_plan": {{
    "stocks_to_trade": 2,
    "capital_to_deploy": "₹60,000",
    "capital_in_cash": "₹40,000",
    "deploy_pct": 60,
    "strategy_note": "1 sentence on overall approach today"
  }},
  "trade_plans": [
    {{
      "rank": 1,
      "ticker": "STOCK",
      "action": "BUY" or "WAIT" or "AVOID",
      "conviction": "HIGH" or "MEDIUM" or "LOW",
      "reasoning": "2-3 sentences why — grounded in data only",
      "entry_price": "₹xxx",
      "stop_loss": "₹xxx",
      "target_1": "₹xxx",
      "target_2": "₹xxx",
      "holding_period": "Intraday" or "1-3 Days" or "1-2 Weeks" or "Swing 2-4 Weeks",
      "capital_to_invest": "₹xx,xxx",
      "capital_pct": 20,
      "qty": 10,
      "expected_profit_t1": "₹x,xxx",
      "expected_profit_t2": "₹x,xxx",
      "max_loss": "₹x,xxx",
      "risk_reward_note": "If SL hits you lose ₹X, if T1 hits you gain ₹Y"
    }}
  ],
  "risk_warning": "1 sentence key risk for today",
  "best_pick_summary": "In 1 sentence, which stock would you bet on and why"
}}

Scan Data:
{json.dumps(payload, default=str)}"""


def _build_no_breakout_prompt(payload: dict, capital: int = 100000) -> str:
    """Prompt when NO breakouts found — bearish/flat market."""
    cap_fmt = f"₹{capital:,}"
    return f"""The NSE breakout scan found 0 breakouts today. All sectors are weak. The market is not giving buy signals right now.
The trader's available capital is {cap_fmt}. Only suggest watchlist stocks whose per-share price the trader can afford (at least 1 share within budget).

But a trader still needs guidance. Analyze the data and return a JSON object with:

{{
  "market_brief": "3-5 sentences explaining WHY no breakouts were found — what's happening in the market. Be specific about index levels and sector weakness.",
  "overall_bias": "BEARISH" or "NEUTRAL",
  "portfolio_plan": {{
    "stocks_to_trade": 0,
    "capital_to_deploy": "₹0",
    "capital_in_cash": "{cap_fmt}",
    "deploy_pct": 0,
    "strategy_note": "Clear instruction — sit in cash, wait for X condition"
  }},
  "trade_plans": [],
  "watchlist": [
    {{
      "ticker": "STOCK from sector data",
      "sector": "sector name",
      "why_watch": "1-2 sentences — what signal to wait for before entering",
      "trigger_level": "₹xxx — price level that would signal a buy"
    }}
  ],
  "re_entry_conditions": "2-3 sentences — what market conditions must change before deploying capital. Be specific about Nifty levels, sector rotation signs, volume patterns.",
  "risk_warning": "1 sentence — key risk if someone tries to buy in this market",
  "best_pick_summary": "Even in a weak market, name 1 stock from the strongest sector to watch and the exact condition to enter"
}}

Scan Data:
{json.dumps(payload, default=str)}"""


def _parse_ai_response(raw: str) -> Optional[dict]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error(f"AI response was not valid JSON: {raw[:200]}")
        return {"market_brief": raw, "trade_plans": [], "watchlist": [],
                "risk_warning": "", "overall_bias": "NEUTRAL",
                "portfolio_plan": None, "best_pick_summary": ""}


def is_configured() -> bool:
    """Check if Azure OpenAI credentials are present."""
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))
