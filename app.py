"""
NSE Breakout Trading Dashboard — Flask Application
AI-powered breakout conviction engine. Hard data only, no noise.
"""

import logging
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask.json.provider import DefaultJSONProvider
from screener import screen_stocks
from data_fetcher import clear_cache
from data_sources import MultiSourceManager
from ai_analysis import generate_ai_analysis, is_configured as ai_configured


class NumpyJSONProvider(DefaultJSONProvider):
    """Handle numpy types that stdlib json can't serialize."""
    @staticmethod
    def default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
msm = MultiSourceManager()

# Cache last scan results for AI analysis endpoint
_last_scan = None


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/scan", methods=["GET"])
def scan():
    """Run breakout scan with conviction scoring."""
    global _last_scan
    try:
        results = screen_stocks()
        _last_scan = results
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        logging.error(f"Scan error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/refresh", methods=["POST"])
def refresh():
    """Clear cache and re-scan."""
    global _last_scan
    clear_cache()
    try:
        results = screen_stocks()
        _last_scan = results
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        logging.error(f"Refresh error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/ai-analysis", methods=["GET"])
def ai_analysis():
    """Generate AI-powered analysis from last scan results."""
    if not ai_configured():
        return jsonify({"status": "error", "message": "Azure OpenAI not configured"}), 503
    if _last_scan is None:
        return jsonify({"status": "error", "message": "Run a scan first"}), 400
    try:
        capital = request.args.get('capital', '100000')
        try:
            capital = max(500, int(float(capital)))
        except (ValueError, TypeError):
            capital = 100000
        analysis = generate_ai_analysis(_last_scan, capital=capital)
        if analysis is None:
            return jsonify({"status": "error", "message": "AI analysis failed"}), 500
        return jsonify({"status": "success", "data": analysis})
    except Exception as e:
        logging.error(f"AI analysis error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/market-overview", methods=["GET"])
def market_overview():
    """Index levels, OI/PCR, advance-decline — hard data only."""
    try:
        overview = msm.get_market_overview()
        return jsonify({"status": "success", "data": overview})
    except Exception as e:
        logging.error(f"Market overview error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/quote/<symbol>", methods=["GET"])
def quote(symbol):
    """Get NSE quote with delivery % for a single stock."""
    try:
        q = msm.get_enriched_quote(symbol.upper())
        if q:
            return jsonify({"status": "success", "data": q})
        return jsonify({"status": "error", "message": "Quote not available"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/sources", methods=["GET"])
def data_sources():
    """List active data sources."""
    sources = msm._get_active_sources()
    return jsonify({
        "status": "success",
        "data": {"active_sources": sources},
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
