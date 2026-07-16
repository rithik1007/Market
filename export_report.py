"""
Export Module — Generate PDF and Excel reports from scan/AI data.
"""

import io
import csv
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def export_scan_csv(scan_data: dict) -> str:
    """Export scan results to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Rank", "Ticker", "Sector", "Price", "Change%", "Pattern",
        "Conviction", "Verdict", "Entry", "Stop Loss", "Target 1",
        "Target 2", "Target 3", "R:R", "RSI", "Volume Ratio",
        "Above 50DMA", "Above 200DMA", "MACD", "Trend",
        "Institutional Score", "Fake Breakout Risk",
    ])

    picks = scan_data.get("top_picks", []) + scan_data.get("all_candidates", [])
    seen = set()
    rank = 0

    for p in picks:
        ticker = p.get("ticker", "")
        if ticker in seen:
            continue
        seen.add(ticker)
        rank += 1

        tl = p.get("trade_levels", {})
        writer.writerow([
            rank,
            ticker,
            p.get("sector", ""),
            p.get("close", ""),
            p.get("change_pct", ""),
            p.get("pattern_name", ""),
            p.get("conviction_score", ""),
            p.get("conviction_verdict", ""),
            tl.get("entry_zone", ""),
            tl.get("stop_loss", ""),
            tl.get("target_1", ""),
            tl.get("target_2", ""),
            tl.get("target_3", ""),
            tl.get("risk_reward", ""),
            p.get("rsi", ""),
            p.get("volume_ratio", ""),
            "Yes" if p.get("above_50dma") else "No",
            "Yes" if p.get("above_200dma") else "No",
            "Bullish" if p.get("macd_bullish") else "Bearish",
            p.get("trend", ""),
            p.get("institutional", {}).get("score", ""),
            f"{p.get('fake_breakout', {}).get('confidence', 0)}%",
        ])

    return output.getvalue()


def export_ai_csv(ai_data: dict) -> str:
    """Export AI recommendations to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Rank", "Ticker", "Action", "Conviction", "Entry Price",
        "Stop Loss", "Target 1", "Target 2", "Qty", "Capital",
        "Holding Period", "Entry Timing", "Exit Timing",
        "Expected Profit T1", "Expected Profit T2", "Max Loss",
        "Reasoning",
    ])

    for p in ai_data.get("trade_plans", []):
        writer.writerow([
            p.get("rank", ""),
            p.get("ticker", ""),
            p.get("action", ""),
            p.get("conviction", ""),
            p.get("entry_price", ""),
            p.get("stop_loss", ""),
            p.get("target_1", ""),
            p.get("target_2", ""),
            p.get("qty", ""),
            p.get("capital_to_invest", ""),
            p.get("holding_period", ""),
            p.get("entry_timing", ""),
            p.get("exit_timing", ""),
            p.get("expected_profit_t1", ""),
            p.get("expected_profit_t2", ""),
            p.get("max_loss", ""),
            p.get("reasoning", ""),
        ])

    # Watchlist
    if ai_data.get("watchlist"):
        writer.writerow([])
        writer.writerow(["--- WATCHLIST ---"])
        writer.writerow(["Ticker", "Sector", "Current Price", "Trigger Level",
                          "Why Watch", "Qty", "Investment"])
        for w in ai_data["watchlist"]:
            writer.writerow([
                w.get("ticker", ""),
                w.get("sector", ""),
                w.get("current_price", ""),
                w.get("trigger_level", ""),
                w.get("why_watch", ""),
                w.get("qty_at_trigger", ""),
                w.get("investment_amount", ""),
            ])

    return output.getvalue()


def export_performance_csv(perf_data: dict) -> str:
    """Export performance stats to CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Summary
    writer.writerow(["NSE Breakout Scanner — Performance Report"])
    writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"])
    writer.writerow([])
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Total Picks", perf_data.get("total_picks", 0)])
    writer.writerow(["Wins (Target Hit)", perf_data.get("wins", 0)])
    writer.writerow(["Losses (SL Hit)", perf_data.get("losses", 0)])
    writer.writerow(["Active", perf_data.get("active", 0)])
    writer.writerow(["Win Rate", f"{perf_data.get('win_rate', 0)}%"])
    writer.writerow(["Avg Return", f"{perf_data.get('avg_return', 0)}%"])
    writer.writerow(["Sharpe Ratio", perf_data.get("sharpe_ratio", 0)])
    writer.writerow(["Max Drawdown", f"{perf_data.get('max_drawdown', 0)}%"])
    writer.writerow(["Best Streak", perf_data.get("best_streak", 0)])
    writer.writerow(["Worst Streak", perf_data.get("worst_streak", 0)])

    # Best/Worst trade
    best = perf_data.get("best_trade")
    worst = perf_data.get("worst_trade")
    if best:
        writer.writerow([])
        writer.writerow(["Best Trade", f"{best['ticker']} +{best['pnl_pct']}% ({best['date']})"])
    if worst:
        writer.writerow(["Worst Trade", f"{worst['ticker']} {worst['pnl_pct']}% ({worst['date']})"])

    # Conviction breakdown
    writer.writerow([])
    writer.writerow(["--- CONVICTION BREAKDOWN ---"])
    writer.writerow(["Level", "Total", "Wins", "Losses", "Win Rate"])
    for level, stats in perf_data.get("conviction_stats", {}).items():
        writer.writerow([level, stats["total"], stats["wins"], stats["losses"], f"{stats['win_rate']}%"])

    # Sector performance
    writer.writerow([])
    writer.writerow(["--- SECTOR PERFORMANCE ---"])
    writer.writerow(["Sector", "Total Picks", "Wins", "Losses", "Win Rate", "Avg Return"])
    for s in perf_data.get("sector_performance", []):
        writer.writerow([
            s["sector"], s["total_picks"], s["wins"], s["losses"],
            f"{s['win_rate']}%", f"{s['avg_return']}%",
        ])

    return output.getvalue()


def generate_report_html(scan_data: dict, ai_data: dict = None, perf_data: dict = None) -> str:
    """Generate a comprehensive HTML report for PDF export (via browser print)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M IST")

    picks_html = ""
    for i, p in enumerate(scan_data.get("top_picks", [])[:10]):
        tl = p.get("trade_levels", {})
        cv = p.get("conviction_score", 0)
        cv_color = "#10b981" if cv >= 70 else ("#f59e0b" if cv >= 50 else "#ef4444")
        picks_html += f"""
        <tr>
            <td>#{i+1}</td>
            <td><b>{p['ticker']}</b><br><small>{p.get('sector','')}</small></td>
            <td>₹{p['close']}</td>
            <td style="color:{'#10b981' if p.get('change_pct',0)>=0 else '#ef4444'}">{p.get('change_pct',0):+.1f}%</td>
            <td>{p.get('pattern_name','')}</td>
            <td style="color:{cv_color};font-weight:bold">{cv}%</td>
            <td>{tl.get('entry_zone','')}</td>
            <td style="color:#ef4444">{tl.get('stop_loss','')}</td>
            <td style="color:#10b981">{tl.get('target_1','')}</td>
            <td>{tl.get('risk_reward','')}:1</td>
        </tr>"""

    ai_section = ""
    if ai_data:
        plans_html = ""
        for p in ai_data.get("trade_plans", []):
            plans_html += f"""
            <div style="border:1px solid #ddd;border-radius:8px;padding:12px;margin:8px 0">
                <b>#{p.get('rank','')} {p.get('ticker','')}</b> — {p.get('action','')} ({p.get('conviction','')})
                <br>Entry: {p.get('entry_price','')} | SL: {p.get('stop_loss','')} | T1: {p.get('target_1','')} | T2: {p.get('target_2','')}
                <br>Qty: {p.get('qty','')} shares | Invest: {p.get('capital_to_invest','')}
                <br><small>{p.get('reasoning','')}</small>
            </div>"""

        ai_section = f"""
        <h2>🤖 AI Trade Recommendations</h2>
        <p><b>Market Bias:</b> {ai_data.get('overall_bias','')}</p>
        <p>{ai_data.get('market_brief','')}</p>
        <p><b>💡 Top Pick:</b> {ai_data.get('best_pick_summary','')}</p>
        {plans_html}
        <p style="color:#ef4444"><b>⚠ Risk:</b> {ai_data.get('risk_warning','')}</p>
        """

    perf_section = ""
    if perf_data and perf_data.get("total_picks", 0) > 0:
        perf_section = f"""
        <h2>📈 Performance Summary</h2>
        <table><tr>
            <td><b>Win Rate:</b> {perf_data.get('win_rate',0)}%</td>
            <td><b>Avg Return:</b> {perf_data.get('avg_return',0)}%</td>
            <td><b>Sharpe:</b> {perf_data.get('sharpe_ratio',0)}</td>
            <td><b>Max DD:</b> {perf_data.get('max_drawdown',0)}%</td>
        </tr></table>
        """

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>NSE Breakout Report — {now}</title>
<style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #333; }}
    h1 {{ color: #6366f1; border-bottom: 2px solid #6366f1; padding-bottom: 8px; }}
    h2 {{ color: #444; margin-top: 30px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }}
    th {{ background: #f1f5f9; padding: 8px; text-align: left; border-bottom: 2px solid #ddd; }}
    td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
    small {{ color: #888; }}
    .disclaimer {{ font-size: 11px; color: #999; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 12px; }}
    @media print {{ body {{ font-size: 12px; }} }}
</style>
</head><body>
<h1>📊 NSE Breakout Scanner Report</h1>
<p>{now} | Scanned: {scan_data.get('total_scanned',0)} stocks | Breakouts: {scan_data.get('breakouts_found',0)}</p>

<h2>🏆 Top Breakout Picks</h2>
<table>
    <tr><th>#</th><th>Stock</th><th>Price</th><th>Change</th><th>Pattern</th><th>Conviction</th><th>Entry</th><th>SL</th><th>Target</th><th>R:R</th></tr>
    {picks_html}
</table>

{ai_section}
{perf_section}

<div class="disclaimer">
    <b>Disclaimer:</b> This report is generated by an automated system for informational purposes only.
    It does not constitute financial advice. Past performance is not indicative of future results.
    Always consult a qualified financial advisor before making investment decisions.
    The creator of this tool is not SEBI registered and is not liable for any trading losses.
</div>
</body></html>"""
