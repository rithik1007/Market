"""
NSE Breakout Trading Dashboard — Flask Application
AI-powered breakout conviction engine. Hard data only, no noise.
"""

import logging
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask.json.provider import DefaultJSONProvider
from screener import screen_stocks
from data_fetcher import clear_cache, fetch_stock_data
from data_sources import MultiSourceManager
from ai_analysis import generate_ai_analysis, is_configured as ai_configured, analyze_single_stock
from analyzer import compute_support_resistance
from nse_stocks import SECTOR_STOCKS, get_stock_sector
from history import save_ai_result, get_history, get_history_tickers


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
        timeframe = request.args.get('timeframe', 'Intraday')
        allowed_timeframes = ['Intraday', '1-2 Days', '3-5 Days', '1 Week', '2 Weeks', '1 Month']
        if timeframe not in allowed_timeframes:
            timeframe = 'Intraday'
        analysis = generate_ai_analysis(_last_scan, capital=capital, timeframe=timeframe)
        if analysis is None:
            return jsonify({"status": "error", "message": "AI analysis failed"}), 500
        # Auto-save to history
        try:
            save_ai_result(analysis, capital, {
                "breakouts_found": _last_scan.get("breakouts_found", 0),
                "total_scanned": _last_scan.get("total_scanned", 0),
            }, timeframe=timeframe)
        except Exception as he:
            logging.warning(f"History save failed: {he}")
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


@app.route("/api/history", methods=["GET"])
def history():
    """Return saved AI recommendation history."""
    try:
        entries = get_history()
        return jsonify({"status": "success", "data": entries})
    except Exception as e:
        logging.error(f"History error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history/performance", methods=["GET"])
def history_performance():
    """Fetch current prices for all historically recommended tickers."""
    try:
        tickers = get_history_tickers()
        prices = {}
        for t in tickers:
            try:
                df = fetch_stock_data(t + ".NS", period_days=10)
                if df is not None and len(df) > 0:
                    prices[t] = round(float(df["Close"].iloc[-1]), 2)
            except Exception:
                pass
        return jsonify({"status": "success", "data": prices})
    except Exception as e:
        logging.error(f"Performance error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


def _compute_indicators_flexible(df):
    """Compute indicators on whatever data we have — no minimum rows required."""
    import pandas as pd
    df = df.copy()
    n = len(df)
    # Moving averages — only compute if enough rows
    if n >= 20:
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["BB_Mid"] = df["SMA_20"]
        bb_std = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
        df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std
        df["Vol_SMA"] = df["Volume"].rolling(20).mean()
    if n >= 50:
        df["SMA_50"] = df["Close"].rolling(50).mean()
    if n >= 200:
        df["SMA_200"] = df["Close"].rolling(200).mean()
    # RSI
    if n >= 14:
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    if n >= 26:
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    # ATR
    if n >= 14:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()
    # VWAP
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


@app.route("/api/analyze-stock", methods=["GET"])
def analyze_stock():
    """AI-powered intraday/swing analysis for a single stock."""
    if not ai_configured():
        return jsonify({"status": "error", "message": "Azure OpenAI not configured"}), 503
    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"status": "error", "message": "Provide ?symbol=STOCKNAME"}), 400

    try:
        yf_symbol = symbol + ".NS"
        df = fetch_stock_data(yf_symbol, period_days=365)

        # If no data, try yfinance search to find the right ticker
        if df is None or len(df) < 30:
            yf_matches = _search_yf_ticker(symbol)
            for m in yf_matches:
                suggestion = m["ticker"]
                if suggestion != symbol:
                    yf_symbol = suggestion + ".NS"
                    df = fetch_stock_data(yf_symbol, period_days=365)
                    if df is not None and len(df) >= 30:
                        symbol = suggestion
                        break

        if df is None or len(df) < 30:
            # Give helpful error with suggestions from our stock list
            matches = _find_similar_stocks(symbol)
            msg = f"Could not find data for '{symbol}'."
            if matches:
                msg += f" Did you mean: {', '.join(matches[:5])}?"
            else:
                msg += " Try the exact NSE ticker symbol (e.g. TATASTEEL, RELIANCE, INFY)."
            return jsonify({"status": "error", "message": msg}), 404

        # Compute indicators inline — flexible, works with any amount of data
        df = _compute_indicators_flexible(df)
        sr = compute_support_resistance(df)
        last = df.iloc[-1]

        def _safe(col, default=0):
            v = last.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return round(float(v), 2)

        stock_data = {
            "ticker": symbol,
            "close": round(float(last["Close"]), 2),
            "open": round(float(last["Open"]), 2),
            "high": round(float(last["High"]), 2),
            "low": round(float(last["Low"]), 2),
            "volume": int(last["Volume"]),
            "prev_close": round(float(df.iloc[-2]["Close"]), 2) if len(df) > 1 else None,
            "change_pct": round(float((last["Close"] - df.iloc[-2]["Close"]) / df.iloc[-2]["Close"] * 100), 2) if len(df) > 1 else 0,
            "rsi": _safe("RSI"),
            "macd": _safe("MACD"),
            "macd_signal": _safe("MACD_Signal"),
            "macd_hist": _safe("MACD_Hist"),
            "sma_20": _safe("SMA_20"),
            "sma_50": _safe("SMA_50"),
            "sma_200": _safe("SMA_200"),
            "ema_20": _safe("EMA_20"),
            "upper_band": _safe("BB_Upper"),
            "lower_band": _safe("BB_Lower"),
            "atr": _safe("ATR"),
            "vwap": _safe("VWAP"),
            "volume_ratio": round(float(last["Volume"] / df["Volume"].rolling(20).mean().iloc[-1]), 2) if df["Volume"].rolling(20).mean().iloc[-1] > 0 else 1.0,
            "above_50dma": bool(last["Close"] > _safe("SMA_50")) if _safe("SMA_50") else None,
            "above_200dma": bool(last["Close"] > _safe("SMA_200")) if _safe("SMA_200") else None,
            "support_levels": sr.get("support", []),
            "resistance_levels": sr.get("resistance", []),
            "5d_high": round(float(df["High"].tail(5).max()), 2),
            "5d_low": round(float(df["Low"].tail(5).min()), 2),
            "20d_high": round(float(df["High"].tail(20).max()), 2),
            "20d_low": round(float(df["Low"].tail(20).min()), 2),
        }

        analysis = analyze_single_stock(stock_data)
        if analysis is None:
            return jsonify({"status": "error", "message": "AI analysis failed"}), 500
        # Include raw technicals so frontend can show them
        analysis["technicals"] = stock_data
        return jsonify({"status": "success", "data": analysis})
    except Exception as e:
        logging.error(f"Analyze stock error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# ──────────────────────────────────────────────
# Stock search helpers
# ──────────────────────────────────────────────
_ALL_STOCKS = []
for _stocks in SECTOR_STOCKS.values():
    _ALL_STOCKS.extend(_stocks)


def _find_similar_stocks(query):
    """Find stocks from our universe that match the query (fuzzy)."""
    q = query.upper().replace(" ", "")
    matches = []
    for s in _ALL_STOCKS:
        if q in s or s in q:
            matches.append(s)
    if not matches:
        for s in _ALL_STOCKS:
            if q[:3] in s or s[:3] in q:
                matches.append(s)
    return matches[:8]


def _search_yf_ticker(query):
    """Use yfinance Search to find NSE tickers matching query."""
    try:
        from yfinance import Search
        results = Search(query).quotes
        matches = []
        if isinstance(results, list):
            for r in results:
                sym = r.get("symbol", "")
                if sym.endswith(".NS"):
                    ticker = sym.replace(".NS", "")
                    name = r.get("shortname", r.get("longname", ""))
                    matches.append({"ticker": ticker, "name": name})
        return matches
    except Exception:
        return []


@app.route("/api/stock-search", methods=["GET"])
def stock_search():
    """Search for stocks by name/ticker — local list + Yahoo Finance for full NSE coverage."""
    q = request.args.get("q", "").strip().upper()
    if len(q) < 1:
        return jsonify({"status": "success", "data": []})

    seen = set()
    results = []

    # 1. Instant local matches (our 150 curated stocks)
    for sector, stocks in SECTOR_STOCKS.items():
        for s in stocks:
            if q in s:
                results.append({"ticker": s, "sector": sector, "name": ""})
                seen.add(s)

    # 2. Yahoo Finance search (covers ALL NSE-listed stocks)
    if len(q) >= 2:
        yf_matches = _search_yf_ticker(q)
        for m in yf_matches:
            t = m["ticker"]
            if t not in seen:
                results.append({"ticker": t, "sector": get_stock_sector(t), "name": m["name"]})
                seen.add(t)

    # 3. Direct ticker probe — if user typed an exact ticker not found above
    if not results and len(q) >= 3:
        try:
            import yfinance as yf
            info = yf.Ticker(q + ".NS").fast_info
            if hasattr(info, "last_price") and info.last_price:
                results.append({"ticker": q, "sector": "NSE", "name": f"₹{info.last_price:.2f}"})
        except Exception:
            pass

    return jsonify({"status": "success", "data": results[:12]})


if __name__ == "__main__":
    import os
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    app.run(debug=debug, host="0.0.0.0", port=port)
