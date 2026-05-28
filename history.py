"""
History Module — Persist daily AI recommendations using ChromaDB.
Tracks picks over time so users can review past performance.
Data survives restarts via ChromaDB's persistent storage.
"""

import os
import json
import logging
from datetime import datetime, date
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "ai_recommendations"

_client = None
_collection = None


def _get_collection():
    """Lazy-init ChromaDB persistent client and collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def save_ai_result(ai_data: dict, capital: int, scan_summary: dict, timeframe: str = "Intraday"):
    """Save an AI analysis result to ChromaDB. One entry per day (upserts same day)."""
    today = date.today().isoformat()
    collection = _get_collection()

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

    # Use date as the unique ID — upsert overwrites same-day entries
    doc_text = json.dumps(entry, ensure_ascii=False)
    collection.upsert(
        ids=[today],
        documents=[doc_text],
        metadatas=[{
            "date": today,
            "capital": capital,
            "timeframe": timeframe,
            "overall_bias": ai_data.get("overall_bias", ""),
            "breakouts_found": scan_summary.get("breakouts_found", 0),
            "total_scanned": scan_summary.get("total_scanned", 0),
        }],
    )
    logger.info(f"Saved AI history to ChromaDB for {today} ({len(picks)} picks)")


def get_history() -> list:
    """Return all history entries, newest first."""
    collection = _get_collection()
    try:
        results = collection.get(include=["documents", "metadatas"])
        entries = []
        for doc in (results.get("documents") or []):
            try:
                entries.append(json.loads(doc))
            except (json.JSONDecodeError, TypeError):
                continue
        # Sort newest first
        entries.sort(key=lambda e: e.get("date", ""), reverse=True)
        return entries
    except Exception as e:
        logger.error(f"ChromaDB get_history error: {e}")
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
