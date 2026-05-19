"""
History Module — Persist daily AI recommendations to a JSON file.
Tracks picks over time so users can review past performance.
"""

import os
import json
import logging
from datetime import datetime, date
from typing import Optional

logger = logging.getLogger(__name__)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")


def _load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.warning("Corrupted history file — starting fresh")
        return []


def _save_history(entries: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def save_ai_result(ai_data: dict, capital: int, scan_summary: dict):
    """Save an AI analysis result to history. One entry per day (overwrites same day)."""
    today = date.today().isoformat()
    entries = _load_history()

    # Extract picks from AI data
    picks = []
    for plan in ai_data.get("trade_plans", []):
        picks.append({
            "ticker": plan.get("ticker", ""),
            "action": plan.get("action", ""),
            "entry_price": plan.get("entry_price", ""),
            "stop_loss": plan.get("stop_loss", ""),
            "target_1": plan.get("target_1", ""),
            "target_2": plan.get("target_2", ""),
            "conviction": plan.get("conviction", ""),
            "reasoning": plan.get("reasoning", ""),
            "qty": plan.get("qty", 0),
            "capital_to_invest": plan.get("capital_to_invest", ""),
        })
    for w in ai_data.get("watchlist", []):
        picks.append({
            "ticker": w.get("ticker", ""),
            "action": "WATCH",
            "entry_price": w.get("trigger_level", w.get("current_price", "")),
            "stop_loss": "",
            "target_1": "",
            "target_2": "",
            "conviction": "",
            "reasoning": w.get("why_watch", ""),
            "qty": w.get("qty_at_trigger", 0),
            "capital_to_invest": w.get("investment_amount", ""),
        })

    entry = {
        "date": today,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "capital": capital,
        "overall_bias": ai_data.get("overall_bias", ""),
        "market_brief": ai_data.get("market_brief", ""),
        "best_pick_summary": ai_data.get("best_pick_summary", ""),
        "portfolio_plan": ai_data.get("portfolio_plan"),
        "picks": picks,
        "breakouts_found": scan_summary.get("breakouts_found", 0),
        "total_scanned": scan_summary.get("total_scanned", 0),
    }

    # Replace existing entry for today, or append
    replaced = False
    for i, e in enumerate(entries):
        if e.get("date") == today:
            entries[i] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)

    # Keep last 90 days
    entries = entries[-90:]
    _save_history(entries)
    logger.info(f"Saved AI history for {today} ({len(picks)} picks)")


def get_history() -> list:
    """Return all history entries, newest first."""
    entries = _load_history()
    entries.reverse()
    return entries


def get_history_tickers() -> list:
    """Return unique tickers from all history entries for performance lookup."""
    entries = _load_history()
    tickers = set()
    for e in entries:
        for p in e.get("picks", []):
            t = p.get("ticker", "")
            if t:
                tickers.add(t)
    return list(tickers)
