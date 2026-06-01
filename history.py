"""
History Module — Persist daily AI recommendations using SQLite.
Tracks picks over time so users can review past performance.
Data survives restarts via SQLite's persistent file storage.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, date
from typing import Optional

logger = logging.getLogger(__name__)

DB_FILE = os.path.join(os.path.dirname(__file__), "history.db")

_conn = None


def _get_conn():
    """Lazy-init SQLite connection and ensure table exists."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_history (
                date TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)
        _conn.commit()
    return _conn


def save_ai_result(ai_data: dict, capital: int, scan_summary: dict, timeframe: str = "Intraday"):
    """Save an AI analysis result to SQLite. One entry per day (upserts same day)."""
    today = date.today().isoformat()
    conn = _get_conn()

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
        "timeframe": timeframe,
        "overall_bias": ai_data.get("overall_bias", ""),
        "market_brief": ai_data.get("market_brief", ""),
        "best_pick_summary": ai_data.get("best_pick_summary", ""),
        "portfolio_plan": ai_data.get("portfolio_plan"),
        "picks": picks,
        "breakouts_found": scan_summary.get("breakouts_found", 0),
        "total_scanned": scan_summary.get("total_scanned", 0),
    }

    doc_text = json.dumps(entry, ensure_ascii=False)
    conn.execute(
        "INSERT OR REPLACE INTO ai_history (date, data) VALUES (?, ?)",
        (today, doc_text),
    )
    conn.commit()
    logger.info(f"Saved AI history to SQLite for {today} ({len(picks)} picks)")


def get_history() -> list:
    """Return all history entries, newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT data FROM ai_history ORDER BY date DESC"
        ).fetchall()
        entries = []
        for (doc,) in rows:
            try:
                entries.append(json.loads(doc))
            except (json.JSONDecodeError, TypeError):
                continue
        return entries
    except Exception as e:
        logger.error(f"SQLite get_history error: {e}")
        return []


def get_history_tickers() -> list:
    """Return unique tickers from all history entries for performance lookup."""
    entries = get_history()
    tickers = set()
    for e in entries:
        for p in e.get("picks", []):
            t = p.get("ticker", "")
            if t:
                tickers.add(t)
    return list(tickers)
