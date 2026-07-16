"""
Scheduler — Daily automated scan and email alert at a configured time.
Runs as a background thread inside the Flask app.
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

_scheduler_thread = None
_scheduler_running = False


def _get_scheduled_time():
    """Return today's scheduled time in IST (default 09:10)."""
    import os
    hour = int(os.getenv("ALERT_SCHEDULE_HOUR", "9"))
    minute = int(os.getenv("ALERT_SCHEDULE_MINUTE", "10"))
    now_ist = datetime.now(IST)
    scheduled = now_ist.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return scheduled


def _seconds_until_next_run():
    """Calculate seconds until next scheduled run."""
    now = datetime.now(IST)
    target = _get_scheduled_time()
    if now >= target:
        # Already passed today, schedule for tomorrow
        target += timedelta(days=1)
    delta = (target - now).total_seconds()
    return max(0, delta)


def _run_daily_alert():
    """Run the scan and send email alert."""
    from screener import screen_stocks
    from alerts import send_breakout_alerts, send_ai_alerts, is_email_configured
    from ai_analysis import generate_ai_analysis, is_configured as ai_configured
    from history import save_ai_result

    if not is_email_configured():
        logger.info("Scheduler: Email not configured, skipping")
        return

    logger.info("Scheduler: Running daily scan at %s IST",
                datetime.now(IST).strftime("%H:%M"))

    try:
        results = screen_stocks()
        top_picks = results.get("top_picks", [])

        if top_picks:
            send_breakout_alerts(top_picks)
            logger.info("Scheduler: Sent breakout alert with %d picks", len(top_picks))

        # Also run AI analysis if configured
        if ai_configured() and results.get("breakouts_found", 0) > 0:
            analysis = generate_ai_analysis(results, capital=100000, timeframe="1 Week")
            if analysis:
                send_ai_alerts(analysis)
                try:
                    save_ai_result(analysis, 100000, {
                        "breakouts_found": results.get("breakouts_found", 0),
                        "total_scanned": results.get("total_scanned", 0),
                    }, timeframe="1 Week")
                except Exception:
                    pass
                logger.info("Scheduler: Sent AI alert")

    except Exception as e:
        logger.error("Scheduler: Daily alert failed: %s", e, exc_info=True)


def _scheduler_loop():
    """Background loop that sleeps until scheduled time, runs, repeats."""
    global _scheduler_running
    logger.info("Scheduler: Started — daily alert at %s IST",
                _get_scheduled_time().strftime("%H:%M"))

    while _scheduler_running:
        wait = _seconds_until_next_run()
        logger.info("Scheduler: Next run in %.0f minutes (%.1f hours)",
                     wait / 60, wait / 3600)

        # Sleep in 60s chunks so we can stop cleanly
        slept = 0
        while slept < wait and _scheduler_running:
            time.sleep(min(60, wait - slept))
            slept += 60

        if _scheduler_running:
            _run_daily_alert()
            # Sleep 61 seconds to avoid double-trigger
            time.sleep(61)


def start_scheduler():
    """Start the daily alert scheduler as a daemon thread."""
    global _scheduler_thread, _scheduler_running

    if _scheduler_running:
        logger.info("Scheduler: Already running")
        return

    _scheduler_running = True
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
    _scheduler_thread.start()
    logger.info("Scheduler: Background thread started")


def stop_scheduler():
    """Stop the scheduler."""
    global _scheduler_running
    _scheduler_running = False
    logger.info("Scheduler: Stopped")


def get_scheduler_status():
    """Return scheduler status for the API."""
    now_ist = datetime.now(IST)
    next_run = _get_scheduled_time()
    if now_ist >= next_run:
        next_run += timedelta(days=1)

    return {
        "running": _scheduler_running,
        "scheduled_time": _get_scheduled_time().strftime("%H:%M IST"),
        "next_run": next_run.strftime("%Y-%m-%d %H:%M IST"),
        "current_time_ist": now_ist.strftime("%H:%M IST"),
    }
