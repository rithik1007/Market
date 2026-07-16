"""
Alert System — Email and Telegram notifications for breakout alerts.
Sends notifications when stocks trigger breakout conditions.
"""

import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def is_email_configured() -> bool:
    """Check if email alerting is configured."""
    return bool(os.getenv("ALERT_EMAIL_SMTP") and os.getenv("ALERT_EMAIL_FROM"))


def is_telegram_configured() -> bool:
    """Check if Telegram alerting is configured."""
    return bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))


def send_email_alert(subject: str, body_html: str) -> bool:
    """Send an email alert."""
    smtp_host = os.getenv("ALERT_EMAIL_SMTP", "")
    smtp_port = int(os.getenv("ALERT_EMAIL_SMTP_PORT", "587"))
    from_email = os.getenv("ALERT_EMAIL_FROM", "")
    password = os.getenv("ALERT_EMAIL_PASSWORD", "")
    to_email = os.getenv("ALERT_EMAIL_TO", from_email)

    if not smtp_host or not from_email:
        logger.warning("Email not configured")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            if password:
                server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())

        logger.info(f"Email alert sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Email alert failed: {e}")
        return False


def send_telegram_alert(message: str) -> bool:
    """Send a Telegram alert."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        logger.warning("Telegram not configured")
        return False

    try:
        import urllib.request
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = json.dumps({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        logger.info("Telegram alert sent")
        return True
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")
        return False


def format_breakout_alert(picks: list) -> dict:
    """Format breakout picks into alert messages."""
    if not picks:
        return {"subject": "", "email_html": "", "telegram_text": ""}

    now = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    count = len(picks)

    # Email HTML
    rows = ""
    for p in picks[:10]:
        conviction = p.get("conviction_score", 0)
        cv_color = "#10b981" if conviction >= 70 else ("#f59e0b" if conviction >= 50 else "#ef4444")
        rows += f"""
        <tr>
            <td style="padding:8px;border-bottom:1px solid #333;font-weight:bold;color:#e2e8f0">{p['ticker']}</td>
            <td style="padding:8px;border-bottom:1px solid #333;color:#e2e8f0">₹{p['close']}</td>
            <td style="padding:8px;border-bottom:1px solid #333;color:{'#10b981' if p.get('change_pct',0)>=0 else '#ef4444'}">{p.get('change_pct',0):+.1f}%</td>
            <td style="padding:8px;border-bottom:1px solid #333;color:#3b82f6">{p.get('pattern_name','')}</td>
            <td style="padding:8px;border-bottom:1px solid #333;color:{cv_color};font-weight:bold">{conviction}%</td>
            <td style="padding:8px;border-bottom:1px solid #333;color:#e2e8f0">{p.get('trade_levels',{}).get('entry_zone','')}</td>
            <td style="padding:8px;border-bottom:1px solid #333;color:#ef4444">{p.get('trade_levels',{}).get('stop_loss','')}</td>
            <td style="padding:8px;border-bottom:1px solid #333;color:#10b981">{p.get('trade_levels',{}).get('target_1','')}</td>
        </tr>"""

    email_html = f"""
    <div style="font-family:Arial,sans-serif;max-width:800px;margin:0 auto;background:#0a0e17;color:#e2e8f0;padding:24px;border-radius:12px">
        <h1 style="color:#6366f1;margin-bottom:4px">🚨 NSE Breakout Alert</h1>
        <p style="color:#94a3b8;margin-bottom:20px">{now} — {count} breakout(s) detected</p>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <tr style="background:#1e293b">
                <th style="padding:10px;text-align:left;color:#94a3b8">Stock</th>
                <th style="padding:10px;text-align:left;color:#94a3b8">Price</th>
                <th style="padding:10px;text-align:left;color:#94a3b8">Change</th>
                <th style="padding:10px;text-align:left;color:#94a3b8">Pattern</th>
                <th style="padding:10px;text-align:left;color:#94a3b8">Conviction</th>
                <th style="padding:10px;text-align:left;color:#94a3b8">Entry</th>
                <th style="padding:10px;text-align:left;color:#94a3b8">SL</th>
                <th style="padding:10px;text-align:left;color:#94a3b8">Target</th>
            </tr>
            {rows}
        </table>
        <p style="color:#64748b;font-size:11px;margin-top:20px">
            ⚠️ This is not financial advice. Always do your own research before trading.
        </p>
    </div>"""

    # Telegram text
    lines = [f"🚨 <b>NSE Breakout Alert</b> — {now}", f"📊 {count} breakout(s) detected\n"]
    for p in picks[:5]:
        conv = p.get("conviction_score", 0)
        lines.append(
            f"<b>{p['ticker']}</b> ₹{p['close']} ({p.get('change_pct',0):+.1f}%)\n"
            f"  Pattern: {p.get('pattern_name','')}\n"
            f"  Conviction: {conv}% | R:R: {p.get('trade_levels',{}).get('risk_reward','')}:1\n"
            f"  Entry: {p.get('trade_levels',{}).get('entry_zone','')} | SL: {p.get('trade_levels',{}).get('stop_loss','')}\n"
            f"  T1: {p.get('trade_levels',{}).get('target_1','')} | T2: {p.get('trade_levels',{}).get('target_2','')}\n"
        )
    lines.append("\n⚠️ Not financial advice. DYOR.")

    return {
        "subject": f"🚨 {count} NSE Breakout(s) — {now}",
        "email_html": email_html,
        "telegram_text": "\n".join(lines),
    }


def format_ai_alert(ai_data: dict) -> dict:
    """Format AI trade plans into alert messages."""
    if not ai_data:
        return {"subject": "", "email_html": "", "telegram_text": ""}

    now = datetime.now().strftime("%Y-%m-%d %H:%M IST")
    bias = ai_data.get("overall_bias", "NEUTRAL")
    plans = ai_data.get("trade_plans", [])
    best = ai_data.get("best_pick_summary", "")

    # Telegram
    lines = [
        f"🤖 <b>AI Trade Alert</b> — {now}",
        f"Market Bias: <b>{bias}</b>",
        f"💡 {best}\n",
    ]
    for p in plans[:3]:
        lines.append(
            f"#{p.get('rank','')} <b>{p.get('ticker','')}</b> — {p.get('action','')}\n"
            f"  Entry: {p.get('entry_price','')} | SL: {p.get('stop_loss','')}\n"
            f"  T1: {p.get('target_1','')} | T2: {p.get('target_2','')}\n"
            f"  Qty: {p.get('qty','')} | Invest: {p.get('capital_to_invest','')}\n"
        )

    return {
        "subject": f"🤖 AI Trade Alert — {bias} — {now}",
        "email_html": f"<h2>AI Trade Alert</h2><p>{best}</p><p>Bias: {bias}</p>",
        "telegram_text": "\n".join(lines),
    }


def send_breakout_alerts(picks: list) -> dict:
    """Send breakout alerts via all configured channels."""
    alert = format_breakout_alert(picks)
    results = {"email": False, "telegram": False}

    if is_email_configured():
        results["email"] = send_email_alert(alert["subject"], alert["email_html"])
    if is_telegram_configured():
        results["telegram"] = send_telegram_alert(alert["telegram_text"])

    return results


def send_ai_alerts(ai_data: dict) -> dict:
    """Send AI analysis alerts via all configured channels."""
    alert = format_ai_alert(ai_data)
    results = {"email": False, "telegram": False}

    if is_email_configured():
        results["email"] = send_email_alert(alert["subject"], alert["email_html"])
    if is_telegram_configured():
        results["telegram"] = send_telegram_alert(alert["telegram_text"])

    return results


def get_alert_config() -> dict:
    """Return current alert configuration status."""
    return {
        "email_configured": is_email_configured(),
        "email_to": os.getenv("ALERT_EMAIL_TO", os.getenv("ALERT_EMAIL_FROM", "")),
        "telegram_configured": is_telegram_configured(),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", "")[:4] + "***" if os.getenv("TELEGRAM_CHAT_ID") else "",
    }
