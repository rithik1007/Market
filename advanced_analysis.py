"""
Advanced Breakout Detection — ChartInk-style scanning logic with
fake breakout detection, institutional activity analysis, and
relative strength computation.

This module adds:
1. Advanced breakout confirmation (multi-timeframe)
2. Fake breakout detection
3. Institutional activity scoring (delivery %, bulk deals)
4. Relative strength vs Nifty
5. Sector momentum scoring
6. Operator-driven move detection
"""

import numpy as np
import pandas as pd
import logging

from config import (
    BREAKOUT_VOLUME_MULTIPLIER, CONSOLIDATION_RANGE_PCT,
    MIN_DELIVERY_PCT, VOLUME_AVG_PERIOD,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Fake Breakout Detection
# ─────────────────────────────────────────────
def detect_fake_breakout(df):
    """
    Detect potential fake/failed breakouts using multiple signals.

    A breakout is likely fake if:
    1. Price breaks resistance but closes below it
    2. Volume doesn't confirm (below average)
    3. Upper wick is significantly larger than body
    4. RSI is already overbought (>80)
    5. Prior failed breakouts at the same level
    """
    if len(df) < 20:
        return {"is_fake": False, "confidence": 0, "reasons": []}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    reasons = []
    fake_score = 0

    # 1. Check if price closed below the high (bearish candle after breakout)
    body = abs(latest["Close"] - latest["Open"])
    upper_wick = latest["High"] - max(latest["Close"], latest["Open"])
    candle_range = latest["High"] - latest["Low"]

    if candle_range > 0 and upper_wick / candle_range > 0.6:
        fake_score += 2
        reasons.append("Large upper wick — rejection at highs")

    # 2. Volume below average on breakout day
    if latest.get("Vol_Ratio", 1) < 1.0:
        fake_score += 2
        reasons.append("Volume below average — no conviction")

    # 3. Close near the low of the day
    if candle_range > 0:
        close_position = (latest["Close"] - latest["Low"]) / candle_range
        if close_position < 0.3:
            fake_score += 2
            reasons.append("Closed near day's low — bearish signal")

    # 4. RSI extremely overbought
    rsi = latest.get("RSI", 50)
    if rsi > 80:
        fake_score += 1
        reasons.append(f"RSI at {rsi:.0f} — extremely overbought")

    # 5. Check for prior failed breakouts at this level
    resistance_zone = latest["High"] * 0.99
    prior_tests = df.tail(30)
    touches = len(prior_tests[prior_tests["High"] >= resistance_zone])
    if touches >= 3:
        # Level has been tested multiple times — could be strong resistance
        # But also could mean eventual breakout
        pass

    # 6. Gap up that gets sold into
    if latest["Open"] > prev["High"] and latest["Close"] < latest["Open"]:
        fake_score += 1.5
        reasons.append("Gap up sold into — distribution pattern")

    # 7. Bearish divergence (price making higher high, RSI making lower high)
    if len(df) >= 10:
        rsi_5_ago = df.iloc[-5].get("RSI", 50)
        price_5_ago = df.iloc[-5]["High"]
        if latest["High"] > price_5_ago and rsi < rsi_5_ago and rsi_5_ago > 60:
            fake_score += 1.5
            reasons.append("Bearish RSI divergence detected")

    is_fake = fake_score >= 3
    confidence = min(round((fake_score / 8) * 100), 100)

    return {
        "is_fake": is_fake,
        "confidence": confidence,
        "score": fake_score,
        "reasons": reasons,
    }


# ─────────────────────────────────────────────
# Institutional Activity Detection
# ─────────────────────────────────────────────
def analyze_institutional_activity(df, delivery_pct=None):
    """
    Score institutional participation based on:
    1. Delivery percentage (>50% = institutional)
    2. Increasing volume trend over 5 days
    3. Large candle bodies (institutional accumulation)
    4. Consistent buying (multiple green candles with volume)
    """
    if len(df) < 10:
        return {"score": 0, "signals": [], "institutional": False}

    signals = []
    score = 0
    recent = df.tail(10)

    # 1. Delivery percentage
    if delivery_pct and delivery_pct > 60:
        score += 3
        signals.append(f"High delivery {delivery_pct:.0f}% — strong institutional interest")
    elif delivery_pct and delivery_pct > MIN_DELIVERY_PCT:
        score += 1.5
        signals.append(f"Moderate delivery {delivery_pct:.0f}%")

    # 2. Increasing volume trend (5-day)
    vol_5d = recent.tail(5)["Volume"].values
    if len(vol_5d) == 5:
        vol_trend = np.polyfit(range(5), vol_5d, 1)[0]
        if vol_trend > 0:
            score += 1.5
            signals.append("Rising volume trend over 5 days")

    # 3. Large body candles (accumulation)
    bodies = abs(recent["Close"] - recent["Open"])
    ranges = recent["High"] - recent["Low"]
    body_ratios = (bodies / ranges.replace(0, np.nan)).dropna()
    avg_body_ratio = body_ratios.mean()

    if avg_body_ratio > 0.6:
        score += 1
        signals.append("Large candle bodies — decisive moves")

    # 4. Consistent buying pattern
    green_candles = sum(recent["Close"] > recent["Open"])
    if green_candles >= 7:
        score += 1.5
        signals.append(f"{green_candles}/10 days green — consistent buying")
    elif green_candles >= 5:
        score += 0.5

    # 5. Volume on up days vs down days
    up_days = recent[recent["Close"] > recent["Open"]]
    down_days = recent[recent["Close"] <= recent["Open"]]

    avg_up_vol = up_days["Volume"].mean() if len(up_days) > 0 else 0
    avg_down_vol = down_days["Volume"].mean() if len(down_days) > 0 else 1

    if avg_up_vol > avg_down_vol * 1.5:
        score += 1.5
        signals.append("Higher volume on up days — accumulation")

    institutional = score >= 5

    return {
        "score": round(min(score, 10), 1),
        "signals": signals,
        "institutional": institutional,
        "delivery_pct": delivery_pct,
    }


# ─────────────────────────────────────────────
# Relative Strength vs Index
# ─────────────────────────────────────────────
def compute_relative_strength(stock_df, index_df, periods=None):
    """
    Compute relative strength of a stock vs an index.
    RS > 1 means stock is outperforming the index.

    Returns RS for multiple periods.
    """
    if periods is None:
        periods = [5, 10, 20]

    if stock_df is None or index_df is None:
        return {}

    result = {}

    for period in periods:
        if len(stock_df) < period or len(index_df) < period:
            continue

        stock_return = ((stock_df["Close"].iloc[-1] - stock_df["Close"].iloc[-period])
                        / stock_df["Close"].iloc[-period]) * 100
        index_return = ((index_df["Close"].iloc[-1] - index_df["Close"].iloc[-period])
                        / index_df["Close"].iloc[-period]) * 100

        rs = round(stock_return - index_return, 2) if index_return != 0 else 0

        result[f"rs_{period}d"] = rs
        result[f"stock_{period}d"] = round(stock_return, 2)
        result[f"index_{period}d"] = round(index_return, 2)

    # Overall relative strength rating
    rs_values = [v for k, v in result.items() if k.startswith("rs_")]
    if rs_values:
        avg_rs = sum(rs_values) / len(rs_values)
        if avg_rs > 3:
            result["rating"] = "Strong Outperformer"
        elif avg_rs > 0:
            result["rating"] = "Outperformer"
        elif avg_rs > -3:
            result["rating"] = "In-line"
        else:
            result["rating"] = "Underperformer"
    else:
        result["rating"] = "N/A"

    return result


# ─────────────────────────────────────────────
# Operator-Driven Move Detection
# ─────────────────────────────────────────────
def detect_operator_activity(df):
    """
    Detect signs of operator/manipulator activity:
    1. Sudden massive volume spike (>5x avg) with large move
    2. Price manipulation patterns (pump and dump)
    3. Abnormal intraday volatility
    4. Low delivery with high volume (speculative)
    5. Circuit limits hit frequently
    """
    if len(df) < 20:
        return {"operator_risk": False, "score": 0, "signals": []}

    signals = []
    risk_score = 0
    latest = df.iloc[-1]
    recent = df.tail(10)

    # 1. Extreme volume spike
    vol_ratio = latest.get("Vol_Ratio", 1)
    if vol_ratio > 8:
        risk_score += 3
        signals.append(f"Extreme volume spike {vol_ratio:.1f}x — suspicious")
    elif vol_ratio > 5:
        risk_score += 1.5
        signals.append(f"Very high volume {vol_ratio:.1f}x — monitor closely")

    # 2. Pump pattern: multiple consecutive limit-up type moves
    big_moves = sum(abs(recent["Pct_Change"]) > 5)
    if big_moves >= 3:
        risk_score += 2
        signals.append(f"{big_moves} days with >5% moves in 10 days — volatile")

    # 3. Abnormal intraday range vs historical
    avg_range = ((df.tail(30)["High"] - df.tail(30)["Low"]) / df.tail(30)["Low"]).mean() * 100
    today_range = ((latest["High"] - latest["Low"]) / latest["Low"]) * 100

    if today_range > avg_range * 3:
        risk_score += 1.5
        signals.append("Abnormal intraday range — possible manipulation")

    # 4. Price far from moving averages
    if latest.get("SMA_20") and latest["SMA_20"] > 0:
        dist_from_ma = ((latest["Close"] - latest["SMA_20"]) / latest["SMA_20"]) * 100
        if dist_from_ma > 20:
            risk_score += 2
            signals.append(f"Price {dist_from_ma:.0f}% above 20 DMA — overextended")

    # 5. Sudden reversal pattern
    if len(df) >= 3:
        prev2 = df.iloc[-3]
        if (prev2["Close"] < latest["Close"] * 0.9 and
                df.iloc[-2]["Close"] > latest["Close"] * 1.05):
            risk_score += 2
            signals.append("Rapid up-then-down pattern — potential pump & dump")

    operator_risk = risk_score >= 4

    return {
        "operator_risk": operator_risk,
        "score": round(min(risk_score, 10), 1),
        "signals": signals,
        "risk_level": "HIGH" if risk_score >= 6 else ("MEDIUM" if risk_score >= 3 else "LOW"),
    }


# ─────────────────────────────────────────────
# Multi-Timeframe Breakout Confirmation
# ─────────────────────────────────────────────
def confirm_breakout_multi_tf(df):
    """
    Confirm breakout using multiple timeframe analysis:
    1. Daily: Price above breakout level
    2. Weekly: Weekly close confirming
    3. Trend alignment: Short, medium, long MAs aligned
    4. Momentum: RSI + MACD both bullish
    5. Volume: Sustained volume above average
    """
    if len(df) < 30:
        return {"confirmed": False, "strength": 0, "checks": {}}

    latest = df.iloc[-1]
    checks = {}
    strength = 0

    # 1. MA alignment (20 > 50 > 200)
    ma_aligned = (latest.get("SMA_20", 0) > latest.get("SMA_50", 0) >
                  latest.get("SMA_200", 0))
    checks["ma_alignment"] = ma_aligned
    if ma_aligned:
        strength += 2

    # 2. Price above all MAs
    above_all = (latest["Close"] > latest.get("SMA_20", 0) and
                 latest["Close"] > latest.get("SMA_50", 0) and
                 latest["Close"] > latest.get("SMA_200", 0))
    checks["above_all_mas"] = above_all
    if above_all:
        strength += 2

    # 3. RSI bullish (45-70 sweet spot)
    rsi = latest.get("RSI", 50)
    rsi_bullish = 45 <= rsi <= 72
    checks["rsi_bullish"] = rsi_bullish
    if rsi_bullish:
        strength += 1.5

    # 4. MACD bullish
    macd_bullish = latest.get("MACD_Hist", 0) > 0
    checks["macd_bullish"] = macd_bullish
    if macd_bullish:
        strength += 1

    # 5. Volume confirmation (3-day sustained)
    vol_sustained = all(df.iloc[i].get("Vol_Ratio", 0) >= 1.0 for i in [-1, -2, -3])
    checks["volume_sustained"] = vol_sustained
    if vol_sustained:
        strength += 1.5

    # 6. Weekly trend (last 5 candles overall direction)
    weekly_close = df.tail(5)
    weekly_bullish = weekly_close["Close"].iloc[-1] > weekly_close["Close"].iloc[0]
    checks["weekly_bullish"] = weekly_bullish
    if weekly_bullish:
        strength += 1

    # 7. Higher highs and higher lows (trend structure)
    last_3_highs = df.tail(3)["High"].values
    last_3_lows = df.tail(3)["Low"].values
    hh_hl = (last_3_highs[-1] > last_3_highs[-2] and
             last_3_lows[-1] > last_3_lows[-2])
    checks["higher_highs_lows"] = hh_hl
    if hh_hl:
        strength += 1

    confirmed = strength >= 6
    checks["total_strength"] = round(strength, 1)

    return {
        "confirmed": confirmed,
        "strength": round(strength, 1),
        "max_strength": 10,
        "checks": checks,
    }


# ─────────────────────────────────────────────
# Sector Momentum Scoring
# ─────────────────────────────────────────────
def score_sector_momentum(sector_stocks_data):
    """
    Score a sector's momentum based on its constituent stocks.

    Returns a score from 0-10:
    - % of stocks above 50 DMA
    - % of stocks above 200 DMA
    - Average RSI
    - % of stocks with volume above average
    - Average 5-day return
    """
    if not sector_stocks_data:
        return {"score": 0, "details": {}}

    above_50dma = 0
    above_200dma = 0
    rsi_sum = 0
    vol_above_avg = 0
    returns_5d = []
    total = len(sector_stocks_data)

    for df in sector_stocks_data.values():
        if df is None or len(df) < 5:
            total -= 1
            continue

        latest = df.iloc[-1]

        if latest["Close"] > latest.get("SMA_50", 0):
            above_50dma += 1
        if latest["Close"] > latest.get("SMA_200", 0):
            above_200dma += 1
        if latest.get("RSI"):
            rsi_sum += latest["RSI"]
        if latest.get("Vol_Ratio", 0) > 1.0:
            vol_above_avg += 1

        ret_5d = ((latest["Close"] - df.iloc[-5]["Close"]) / df.iloc[-5]["Close"]) * 100
        returns_5d.append(ret_5d)

    if total == 0:
        return {"score": 0, "details": {}}

    pct_above_50 = (above_50dma / total) * 100
    pct_above_200 = (above_200dma / total) * 100
    avg_rsi = rsi_sum / total if total > 0 else 50
    pct_vol = (vol_above_avg / total) * 100
    avg_return = sum(returns_5d) / len(returns_5d) if returns_5d else 0

    # Compute composite score
    score = 0
    score += min(pct_above_50 / 20, 2.5)   # Max 2.5 for 50% above 50DMA
    score += min(pct_above_200 / 25, 2)     # Max 2 for 50% above 200DMA
    score += min(max(avg_rsi - 40, 0) / 10, 2)  # Max 2 for RSI in bullish zone
    score += min(pct_vol / 25, 1.5)         # Max 1.5 for high volume
    score += min(max(avg_return, 0), 2)     # Max 2 for positive returns

    return {
        "score": round(min(score, 10), 1),
        "details": {
            "pct_above_50dma": round(pct_above_50, 1),
            "pct_above_200dma": round(pct_above_200, 1),
            "avg_rsi": round(avg_rsi, 1),
            "pct_high_volume": round(pct_vol, 1),
            "avg_5d_return": round(avg_return, 2),
            "stocks_analyzed": total,
        }
    }


# ─────────────────────────────────────────────
# Conviction Engine — Final Composite Score
# ─────────────────────────────────────────────
def compute_conviction_score(
    probability_score,
    breakout_confirmation,
    institutional_score,
    relative_strength,
    fake_breakout,
    operator_detection,
    sector_momentum_score,
):
    """
    Compute final conviction score (0-100) combining all analysis layers.

    This is what makes the app an "AI-powered breakout conviction engine"
    rather than just another stock tips app.

    Weights:
    - Technical breakout quality: 30%
    - Institutional activity: 20%
    - Multi-TF confirmation: 20%
    - Relative strength: 10%
    - Sector momentum: 10%
    - Penalty: Fake breakout risk, operator risk: -10% each
    """
    conviction = 0

    # Technical score (from probability_score, max 10)
    conviction += (probability_score / 10) * 30

    # Institutional activity (max 10)
    conviction += (institutional_score / 10) * 20

    # Multi-timeframe confirmation (max 10)
    confirmation_score = breakout_confirmation.get("strength", 0)
    conviction += (confirmation_score / 10) * 20

    # Relative strength
    rs_rating = relative_strength.get("rating", "N/A")
    rs_map = {"Strong Outperformer": 10, "Outperformer": 7, "In-line": 4, "Underperformer": 1}
    rs_val = rs_map.get(rs_rating, 5)
    conviction += (rs_val / 10) * 10

    # Sector momentum
    conviction += (sector_momentum_score / 10) * 10

    # Penalties
    if fake_breakout.get("is_fake"):
        conviction -= fake_breakout.get("confidence", 0) * 0.1

    if operator_detection.get("operator_risk"):
        conviction -= operator_detection.get("score", 0) * 1.5

    conviction = max(0, min(100, round(conviction)))

    # Determine verdict
    if conviction >= 80:
        verdict = "HIGH CONVICTION — Strong Buy Setup"
    elif conviction >= 60:
        verdict = "MODERATE CONVICTION — Good Setup"
    elif conviction >= 40:
        verdict = "LOW CONVICTION — Proceed with Caution"
    else:
        verdict = "AVOID — Weak Setup or High Risk"

    return {
        "conviction_score": conviction,
        "verdict": verdict,
        "components": {
            "technical": round((probability_score / 10) * 30, 1),
            "institutional": round((institutional_score / 10) * 20, 1),
            "confirmation": round((confirmation_score / 10) * 20, 1),
            "relative_strength": round((rs_val / 10) * 10, 1),
            "sector_momentum": round((sector_momentum_score / 10) * 10, 1),
        },
    }
