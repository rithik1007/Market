"""
Backtesting Engine — Analyze historical AI recommendation performance.
Computes win rate, avg return, drawdown, Sharpe ratio, and conviction-level stats.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Optional
from history import _get_conn
from data_fetcher import fetch_stock_data

logger = logging.getLogger(__name__)


def get_performance_stats() -> dict:
    """Compute aggregate performance statistics from AI history."""
    conn = _get_conn()
    rows = conn.execute("SELECT data FROM ai_history ORDER BY date DESC").fetchall()

    total_picks = 0
    wins = 0
    losses = 0
    active = 0
    total_return_pct = 0
    returns = []
    conviction_stats = {"HIGH": {"wins": 0, "losses": 0, "total": 0},
                        "MEDIUM": {"wins": 0, "losses": 0, "total": 0},
                        "LOW": {"wins": 0, "losses": 0, "total": 0}}
    sector_stats = {}
    daily_pnl = []
    best_trade = None
    worst_trade = None

    for (doc,) in rows:
        try:
            entry = json.loads(doc)
        except (json.JSONDecodeError, TypeError):
            continue

        date_str = entry.get("date", "")
        day_pnl = 0

        for pick in entry.get("picks", []):
            if pick.get("action") != "BUY":
                continue

            ticker = pick.get("ticker", "")
            if not ticker:
                continue

            entry_price = _parse_price(pick.get("entry_price", ""))
            target_1 = _parse_price(pick.get("target_1", ""))
            stop_loss = _parse_price(pick.get("stop_loss", ""))
            conviction = pick.get("conviction", "MEDIUM").upper()

            if not entry_price:
                continue

            # Fetch current price
            try:
                df = fetch_stock_data(ticker + ".NS", period_days=10)
                if df is None or len(df) == 0:
                    continue
                current_price = float(df["Close"].iloc[-1])
            except Exception:
                continue

            total_picks += 1
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            returns.append(pnl_pct)
            total_return_pct += pnl_pct
            qty = pick.get("qty", 1) or 1
            day_pnl += pnl_pct * qty

            # Determine outcome
            if target_1 and current_price >= target_1:
                wins += 1
                outcome = "win"
            elif stop_loss and current_price <= stop_loss:
                losses += 1
                outcome = "loss"
            else:
                active += 1
                outcome = "active"

            # Conviction tracking
            if conviction in conviction_stats:
                conviction_stats[conviction]["total"] += 1
                if outcome == "win":
                    conviction_stats[conviction]["wins"] += 1
                elif outcome == "loss":
                    conviction_stats[conviction]["losses"] += 1

            # Sector tracking
            sector = pick.get("sector", "Unknown")
            if sector not in sector_stats:
                sector_stats[sector] = {"wins": 0, "losses": 0, "total": 0, "total_return": 0}
            sector_stats[sector]["total"] += 1
            sector_stats[sector]["total_return"] += pnl_pct
            if outcome == "win":
                sector_stats[sector]["wins"] += 1
            elif outcome == "loss":
                sector_stats[sector]["losses"] += 1

            # Best/worst trade
            trade_info = {"ticker": ticker, "pnl_pct": round(pnl_pct, 2),
                          "date": date_str, "conviction": conviction}
            if best_trade is None or pnl_pct > best_trade["pnl_pct"]:
                best_trade = trade_info
            if worst_trade is None or pnl_pct < worst_trade["pnl_pct"]:
                worst_trade = trade_info

        daily_pnl.append({"date": date_str, "pnl": round(day_pnl, 2)})

    # Calculate stats
    win_rate = (wins / total_picks * 100) if total_picks > 0 else 0
    avg_return = (total_return_pct / total_picks) if total_picks > 0 else 0

    # Sharpe ratio (simplified)
    if len(returns) > 1:
        import numpy as np
        returns_arr = np.array(returns)
        sharpe = (returns_arr.mean() / returns_arr.std()) if returns_arr.std() > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    max_drawdown = _compute_max_drawdown(returns)

    # Conviction win rates
    for cv in conviction_stats.values():
        cv["win_rate"] = round((cv["wins"] / cv["total"] * 100) if cv["total"] > 0 else 0, 1)

    # Sector performance
    sector_perf = []
    for sector, stats in sector_stats.items():
        sector_perf.append({
            "sector": sector,
            "total_picks": stats["total"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": round((stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0, 1),
            "avg_return": round((stats["total_return"] / stats["total"]) if stats["total"] > 0 else 0, 2),
        })
    sector_perf.sort(key=lambda x: x["avg_return"], reverse=True)

    # Win/loss streaks
    streak_data = _compute_streaks(returns)

    return {
        "total_picks": total_picks,
        "wins": wins,
        "losses": losses,
        "active": active,
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_drawdown, 2),
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "conviction_stats": conviction_stats,
        "sector_performance": sector_perf,
        "daily_pnl": daily_pnl[-30:],  # Last 30 days
        "total_days_tracked": len(rows),
        **streak_data,
    }


def _parse_price(price_str) -> Optional[float]:
    """Parse price string like '₹2,850' to float."""
    if not price_str:
        return None
    try:
        cleaned = str(price_str).replace("₹", "").replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _compute_max_drawdown(returns: list) -> float:
    """Compute maximum drawdown from a list of returns."""
    if not returns:
        return 0
    cumulative = 0
    peak = 0
    max_dd = 0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_streaks(returns: list) -> dict:
    """Compute winning and losing streaks."""
    if not returns:
        return {"best_streak": 0, "worst_streak": 0, "current_streak": 0}

    current = 0
    best_win = 0
    worst_loss = 0
    current_win = 0
    current_loss = 0

    for r in returns:
        if r >= 0:
            current_win += 1
            current_loss = 0
            best_win = max(best_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            worst_loss = max(worst_loss, current_loss)

    # Current streak direction
    if returns[-1] >= 0:
        current = current_win
    else:
        current = -current_loss

    return {
        "best_streak": best_win,
        "worst_streak": worst_loss,
        "current_streak": current,
    }
