"""
Stock Screener — Filters and ranks stocks based on breakout criteria,
applies strict filtering conditions, and produces the final trading dashboard data.
Enhanced with multi-source data, conviction scoring, and advanced breakout detection.
"""

import logging
from datetime import datetime

from config import (
    MIN_PRICE, MAX_PRICE, MIN_AVG_VOLUME, MAX_INTRADAY_MOVE,
    MIN_RISK_REWARD, TOP_PICKS,
)
from nse_stocks import get_stock_sector, SECTOR_STOCKS
from data_fetcher import fetch_multiple_stocks, fetch_index_data, get_stock_info
from analyzer import (
    compute_indicators, detect_all_patterns, compute_support_resistance,
    calculate_trade_levels, compute_probability_score, get_trend,
    get_rsi_analysis, get_volume_analysis,
)
from advanced_analysis import (
    detect_fake_breakout, analyze_institutional_activity,
    detect_operator_activity, confirm_breakout_multi_tf,
    compute_conviction_score, compute_relative_strength,
    score_sector_momentum,
)
from data_sources import MultiSourceManager

logger = logging.getLogger(__name__)


def screen_stocks(tickers=None):
    """
    Main screening pipeline:
    1. Fetch data for all stocks
    2. Apply technical indicators
    3. Filter based on strict conditions
    4. Detect breakout patterns
    5. Rank by probability score
    6. Produce final dashboard data
    """
    logger.info("Starting stock screening...")

    # Fetch all stock data
    stock_data = fetch_multiple_stocks(tickers)
    if not stock_data:
        logger.error("No stock data fetched")
        return empty_result()

    candidates = []
    rejected = []
    stock_universe = []  # All scanned stocks with basic metrics for AI

    for ticker, df in stock_data.items():
        clean_ticker = ticker.replace(".NS", "")

        # Step 1: Compute indicators
        df = compute_indicators(df)
        if df is None:
            rejected.append({"ticker": clean_ticker, "reason": "Insufficient data for indicators"})
            continue

        latest = df.iloc[-1]
        close = latest["Close"]

        # Collect basic metrics for AI universe (all stocks, pre-filter)
        stock_universe.append({
            "ticker": clean_ticker,
            "sector": get_stock_sector(ticker),
            "close": round(close, 2),
            "change_pct": round(latest.get("Pct_Change", 0), 2),
            "rsi": round(latest["RSI"], 1) if latest["RSI"] is not None else 0,
            "volume_ratio": round(latest.get("Vol_Ratio", 0), 2),
            "above_50dma": bool(close > latest["SMA_50"]),
            "above_200dma": bool(close > latest["SMA_200"]),
            "macd_bullish": bool(latest["MACD_Hist"] > 0),
            "trend": get_trend(df),
        })

        # Step 2: Apply strict filters
        # Price filter
        if close < MIN_PRICE or close > MAX_PRICE:
            rejected.append({"ticker": clean_ticker, "reason": f"Price ₹{close:.0f} outside range"})
            continue

        # Volume/liquidity filter
        avg_vol = latest.get("Vol_SMA", 0)
        if avg_vol < MIN_AVG_VOLUME:
            rejected.append({"ticker": clean_ticker, "reason": "Low liquidity"})
            continue

        # Avoid overextended stocks
        pct_change = latest.get("Pct_Change", 0)
        if abs(pct_change) > MAX_INTRADAY_MOVE:
            rejected.append({"ticker": clean_ticker, "reason": f"Overextended ({pct_change:.1f}%)"})
            continue

        # Prefer stocks above 50 DMA and 200 DMA
        above_50dma = close > latest["SMA_50"]
        above_200dma = close > latest["SMA_200"]

        # Step 3: Detect breakout patterns
        patterns = detect_all_patterns(df)
        if not patterns:
            continue  # No breakout detected, skip

        # Step 4: Calculate trade levels using the strongest pattern
        primary_pattern = patterns[0]
        trade_levels = calculate_trade_levels(df, primary_pattern)

        # Risk-reward filter
        if trade_levels["risk_reward"] < MIN_RISK_REWARD:
            rejected.append({"ticker": clean_ticker, "reason": f"Low R:R ({trade_levels['risk_reward']})"})
            continue

        # Step 5: Compute probability score
        prob_score = compute_probability_score(df, patterns, trade_levels)

        # Step 6: Compute support/resistance
        sr_levels = compute_support_resistance(df)

        # Get sector
        sector = get_stock_sector(ticker)
        trend = get_trend(df)

        # Determine risk factors
        risk_factors = []
        if not above_50dma:
            risk_factors.append("Below 50 DMA")
        if not above_200dma:
            risk_factors.append("Below 200 DMA")
        if latest["RSI"] > 75:
            risk_factors.append("RSI overbought")
        if latest["Vol_Ratio"] < 1.0:
            risk_factors.append("Weak volume")
        if latest["MACD_Hist"] < 0:
            risk_factors.append("MACD bearish")
        if not risk_factors:
            risk_factors.append("Low risk setup")

        # Advanced Analysis — Fake breakout detection
        fake_bo = detect_fake_breakout(df)
        if fake_bo["is_fake"] and fake_bo["confidence"] > 60:
            risk_factors.append(f"⚠ Possible fake breakout ({fake_bo['confidence']}% confidence)")
            for r in fake_bo["reasons"][:2]:
                risk_factors.append(f"  → {r}")

        # Advanced Analysis — Operator detection
        operator_result = detect_operator_activity(df)
        operator_risk = operator_result["operator_risk"]
        if operator_risk:
            risk_factors.append(f"⚠ Operator risk: {operator_result['risk_level']}")
            for s in operator_result["signals"][:2]:
                risk_factors.append(f"  → {s}")

        if not risk_factors:
            risk_factors.append("Low risk setup")

        # Advanced Analysis — Institutional activity
        institutional = analyze_institutional_activity(df)

        # Advanced Analysis — Multi-TF confirmation
        confirmation = confirm_breakout_multi_tf(df)

        # Advanced Analysis — Conviction score
        conviction = compute_conviction_score(
            probability_score=prob_score,
            breakout_confirmation=confirmation,
            institutional_score=institutional["score"],
            relative_strength={"rating": "In-line"},  # Computed at sector level
            fake_breakout=fake_bo,
            operator_detection=operator_result,
            sector_momentum_score=5,  # Default, updated later
        )

        candidate = {
            "ticker": clean_ticker,
            "name": clean_ticker,
            "sector": sector,
            "close": round(close, 2),
            "change_pct": round(pct_change, 2),
            "trend": trend,
            "patterns": patterns,
            "pattern_name": primary_pattern["pattern"],
            "breakout_level": primary_pattern.get("breakout_level", close),
            "volume_confirmed": primary_pattern.get("volume_confirmation", False),
            "trade_levels": trade_levels,
            "rsi": round(latest["RSI"], 1) if not isinstance(latest["RSI"], type(None)) else 0,
            "rsi_analysis": get_rsi_analysis(latest["RSI"]),
            "volume_ratio": round(latest["Vol_Ratio"], 2),
            "volume_analysis": get_volume_analysis(latest["Vol_Ratio"]),
            "sma_50": round(latest["SMA_50"], 2),
            "sma_200": round(latest["SMA_200"], 2),
            "above_50dma": above_50dma,
            "above_200dma": above_200dma,
            "macd_bullish": latest["MACD_Hist"] > 0,
            "support_resistance": sr_levels,
            "probability_score": prob_score,
            "risk_factors": risk_factors,
            "operator_risk": operator_risk,
            # Advanced fields
            "conviction_score": conviction["conviction_score"],
            "conviction_verdict": conviction["verdict"],
            "conviction_components": conviction["components"],
            "fake_breakout": fake_bo,
            "institutional": institutional,
            "confirmation": confirmation,
            "operator_analysis": operator_result,
        }

        candidates.append(candidate)

    # Sort by conviction score (primary) then probability score (secondary)
    candidates.sort(key=lambda x: (x["conviction_score"], x["probability_score"]), reverse=True)

    # Build result
    top_picks = candidates[:TOP_PICKS]

    # Classify picks
    safe_picks = [c for c in candidates
                  if c["conviction_score"] >= 60 and not c["operator_risk"]
                  and not c.get("fake_breakout", {}).get("is_fake")][:3]
    aggressive_picks = [c for c in candidates
                        if c["volume_ratio"] >= 2.0 and c["change_pct"] >= 2][:3]

    # If not enough in specific categories, fill from top picks
    if len(safe_picks) < 3:
        safe_picks = sorted(candidates,
                            key=lambda x: (x["above_200dma"], x["probability_score"]),
                            reverse=True)[:3]
    if len(aggressive_picks) < 3:
        aggressive_picks = sorted(candidates,
                                  key=lambda x: (x["volume_ratio"], x["change_pct"]),
                                  reverse=True)[:3]

    # Stocks to avoid
    avoid_stocks = [r for r in rejected if "Overextended" in r["reason"] or "operator" in r["reason"].lower()][:5]
    # Also add stocks with fake breakouts
    for c in candidates:
        if c.get("fake_breakout", {}).get("is_fake") and c["fake_breakout"]["confidence"] > 50:
            avoid_stocks.append({
                "ticker": c["ticker"],
                "reason": f"Fake breakout detected ({c['fake_breakout']['confidence']}% confidence)",
            })

    # Sector analysis
    sector_analysis = compute_sector_strength(stock_data)

    # Index analysis
    index_analysis = compute_index_sentiment()

    # OI-based market sentiment (hard data — PCR from NSE option chain)
    oi_sentiment = None
    try:
        msm = MultiSourceManager()
        overview = msm.get_market_overview()
        oi_sentiment = overview.get("oi_sentiment") if overview else None
    except Exception as e:
        logger.warning(f"OI sentiment fetch failed: {e}")

    # Add "why this stock" reasoning
    for pick in top_picks:
        pick["why_better"] = generate_why_better(pick, candidates)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "total_scanned": len(stock_data),
        "breakouts_found": len(candidates),
        "top_picks": top_picks,
        "safe_picks": safe_picks,
        "aggressive_picks": aggressive_picks,
        "avoid_stocks": avoid_stocks[:8],
        "sector_analysis": sector_analysis,
        "index_sentiment": index_analysis,
        "all_candidates": candidates,
        "stock_universe": stock_universe,
        "caution_alerts": generate_caution_alerts(candidates, index_analysis),
        "oi_sentiment": oi_sentiment,
    }


def compute_sector_strength(stock_data):
    """Analyze sector rotation and strength."""
    sector_perf = {}

    for ticker, df in stock_data.items():
        sector = get_stock_sector(ticker)
        if sector == "Unknown":
            continue

        if len(df) < 5:
            continue

        pct_5d = ((df["Close"].iloc[-1] - df["Close"].iloc[-5]) / df["Close"].iloc[-5]) * 100

        # Skip if NaN
        if pct_5d != pct_5d:  # NaN check
            continue

        if sector not in sector_perf:
            sector_perf[sector] = []
        sector_perf[sector].append(pct_5d)

    # Average sector performance
    sector_analysis = []
    for sector, perfs in sector_perf.items():
        avg_perf = sum(perfs) / len(perfs)
        sector_analysis.append({
            "sector": sector,
            "avg_5d_return": round(avg_perf, 2),
            "stocks_analyzed": len(perfs),
            "momentum": "Strong" if avg_perf > 2 else ("Moderate" if avg_perf > 0 else "Weak"),
        })

    sector_analysis.sort(key=lambda x: x["avg_5d_return"], reverse=True)
    return sector_analysis


def compute_index_sentiment():
    """Compute market sentiment for Nifty and Bank Nifty."""
    index_data = fetch_index_data()
    sentiments = []

    for name, df in index_data.items():
        if df is None or len(df) < 20:
            continue

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

        sma_20 = df["Close"].rolling(20).mean().iloc[-1]
        sma_50 = df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20

        above_20 = latest["Close"] > sma_20
        above_50 = latest["Close"] > sma_50

        if above_20 and above_50 and change > 0:
            sentiment = "Bullish"
        elif above_20 and change > -0.5:
            sentiment = "Moderately Bullish"
        elif not above_20 and not above_50:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        sentiments.append({
            "index": name,
            "close": round(latest["Close"], 2),
            "change_pct": round(change, 2),
            "sentiment": sentiment,
            "above_20dma": above_20,
            "above_50dma": above_50,
        })

    return sentiments


def generate_why_better(pick, all_candidates):
    """Generate reasoning for why this stock stands out."""
    reasons = []

    if pick["probability_score"] >= 8:
        reasons.append(f"Highest probability score ({pick['probability_score']}/10)")
    if pick["above_50dma"] and pick["above_200dma"]:
        reasons.append("Trading above both 50 & 200 DMA (strong trend)")
    if pick["volume_ratio"] >= 2.0:
        reasons.append(f"Volume surge {pick['volume_ratio']}x average")
    if pick["macd_bullish"]:
        reasons.append("MACD histogram positive (momentum building)")
    if len(pick["patterns"]) > 1:
        reasons.append(f"Multiple breakout patterns ({len(pick['patterns'])})")
    if pick["trade_levels"]["risk_reward"] >= 3:
        reasons.append(f"Excellent R:R of {pick['trade_levels']['risk_reward']}:1")
    if 50 <= pick["rsi"] <= 65:
        reasons.append("RSI in ideal bullish zone")

    if not reasons:
        reasons.append("Clean breakout setup with good technical structure")

    return reasons


def generate_caution_alerts(candidates, index_sentiment):
    """Generate intraday caution alerts."""
    alerts = []

    # Check for bearish index
    for idx in index_sentiment:
        if idx["sentiment"] == "Bearish":
            alerts.append(f"⚠ {idx['index']} is in bearish territory — trade with caution")
        if idx["change_pct"] < -1:
            alerts.append(f"⚠ {idx['index']} down {idx['change_pct']}% — market weakness")

    # Check for overbought candidates
    overbought = [c for c in candidates if c["rsi"] > 75]
    if overbought:
        names = ", ".join([c["ticker"] for c in overbought[:3]])
        alerts.append(f"⚠ RSI overbought: {names} — risk of pullback")

    # Check for operator risk
    operator_stocks = [c for c in candidates if c["operator_risk"]]
    if operator_stocks:
        names = ", ".join([c["ticker"] for c in operator_stocks[:3]])
        alerts.append(f"⚠ Possible operator activity: {names}")

    if not alerts:
        alerts.append("✅ No major caution alerts — market conditions normal")

    return alerts


def empty_result():
    """Return empty result structure."""
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "total_scanned": 0,
        "breakouts_found": 0,
        "top_picks": [],
        "safe_picks": [],
        "aggressive_picks": [],
        "avoid_stocks": [],
        "sector_analysis": [],
        "index_sentiment": [],
        "all_candidates": [],
        "caution_alerts": ["⚠ Unable to fetch market data. Please try again."],
    }
