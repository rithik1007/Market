"""
Watchlist Manager — User-managed watchlists stored in SQLite.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional
from history import _get_conn

logger = logging.getLogger(__name__)


def _ensure_watchlist_table():
    """Create watchlist table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist_stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watchlist_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            added_at TEXT NOT NULL,
            notes TEXT DEFAULT '',
            alert_above REAL DEFAULT NULL,
            alert_below REAL DEFAULT NULL,
            FOREIGN KEY (watchlist_id) REFERENCES watchlists(id),
            UNIQUE(watchlist_id, ticker)
        )
    """)
    conn.commit()


def create_watchlist(name: str) -> dict:
    """Create a new watchlist."""
    _ensure_watchlist_table()
    conn = _get_conn()
    now = datetime.now().isoformat()
    cursor = conn.execute(
        "INSERT INTO watchlists (name, created_at, updated_at) VALUES (?, ?, ?)",
        (name, now, now)
    )
    conn.commit()
    return {"id": cursor.lastrowid, "name": name, "created_at": now, "stocks": []}


def get_watchlists() -> list:
    """Get all watchlists with stock counts."""
    _ensure_watchlist_table()
    conn = _get_conn()
    rows = conn.execute("""
        SELECT w.id, w.name, w.created_at, w.updated_at, COUNT(ws.id) as stock_count
        FROM watchlists w
        LEFT JOIN watchlist_stocks ws ON w.id = ws.watchlist_id
        GROUP BY w.id
        ORDER BY w.updated_at DESC
    """).fetchall()
    return [
        {"id": r[0], "name": r[1], "created_at": r[2], "updated_at": r[3], "stock_count": r[4]}
        for r in rows
    ]


def get_watchlist_stocks(watchlist_id: int) -> list:
    """Get all stocks in a watchlist."""
    _ensure_watchlist_table()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, ticker, added_at, notes, alert_above, alert_below FROM watchlist_stocks WHERE watchlist_id = ?",
        (watchlist_id,)
    ).fetchall()
    return [
        {"id": r[0], "ticker": r[1], "added_at": r[2], "notes": r[3],
         "alert_above": r[4], "alert_below": r[5]}
        for r in rows
    ]


def add_stock_to_watchlist(watchlist_id: int, ticker: str, notes: str = "",
                            alert_above: float = None, alert_below: float = None) -> bool:
    """Add a stock to a watchlist."""
    _ensure_watchlist_table()
    conn = _get_conn()
    now = datetime.now().isoformat()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO watchlist_stocks (watchlist_id, ticker, added_at, notes, alert_above, alert_below) VALUES (?, ?, ?, ?, ?, ?)",
            (watchlist_id, ticker.upper(), now, notes, alert_above, alert_below)
        )
        conn.execute("UPDATE watchlists SET updated_at = ? WHERE id = ?", (now, watchlist_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Add to watchlist failed: {e}")
        return False


def remove_stock_from_watchlist(watchlist_id: int, ticker: str) -> bool:
    """Remove a stock from a watchlist."""
    _ensure_watchlist_table()
    conn = _get_conn()
    try:
        conn.execute(
            "DELETE FROM watchlist_stocks WHERE watchlist_id = ? AND ticker = ?",
            (watchlist_id, ticker.upper())
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Remove from watchlist failed: {e}")
        return False


def delete_watchlist(watchlist_id: int) -> bool:
    """Delete a watchlist and all its stocks."""
    _ensure_watchlist_table()
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM watchlist_stocks WHERE watchlist_id = ?", (watchlist_id,))
        conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Delete watchlist failed: {e}")
        return False


def check_price_alerts(current_prices: dict) -> list:
    """Check if any watchlist stocks have triggered price alerts."""
    _ensure_watchlist_table()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT ws.ticker, ws.alert_above, ws.alert_below, w.name FROM watchlist_stocks ws JOIN watchlists w ON ws.watchlist_id = w.id WHERE ws.alert_above IS NOT NULL OR ws.alert_below IS NOT NULL"
    ).fetchall()

    triggered = []
    for ticker, above, below, wl_name in rows:
        price = current_prices.get(ticker)
        if price is None:
            continue
        if above and price >= above:
            triggered.append({
                "ticker": ticker, "price": price, "alert_type": "above",
                "alert_level": above, "watchlist": wl_name,
                "message": f"🔔 {ticker} crossed above ₹{above:.2f} (now ₹{price:.2f})"
            })
        if below and price <= below:
            triggered.append({
                "ticker": ticker, "price": price, "alert_type": "below",
                "alert_level": below, "watchlist": wl_name,
                "message": f"🔔 {ticker} dropped below ₹{below:.2f} (now ₹{price:.2f})"
            })

    return triggered
