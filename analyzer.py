"""
Technical Analysis Engine — Computes indicators, detects breakout patterns,
and generates trading signals for NSE stocks.
"""

import pandas as pd
import numpy as np
from config import (
    SHORT_MA, MEDIUM_MA, LONG_MA, RSI_PERIOD, VOLUME_AVG_PERIOD,
    CONSOLIDATION_DAYS, ATR_PERIOD, BREAKOUT_VOLUME_MULTIPLIER,
    CONSOLIDATION_RANGE_PCT, MIN_RISK_REWARD,
    SWING_TARGET_1_PCT, SWING_TARGET_2_PCT, SWING_TARGET_3_PCT,
)


def compute_indicators(df):
    """Add all technical indicators to the DataFrame."""
    if df is None or len(df) < LONG_MA:
        return None

    df = df.copy()

    # Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=SHORT_MA).mean()
    df["SMA_50"] = df["Close"].rolling(window=MEDIUM_MA).mean()
    df["SMA_200"] = df["Close"].rolling(window=LONG_MA).mean()
    df["EMA_20"] = df["Close"].ewm(span=SHORT_MA, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Volume Analysis
    df["Vol_SMA"] = df["Volume"].rolling(window=VOLUME_AVG_PERIOD).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA"].replace(0, np.nan)

    # ATR (Average True Range)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=ATR_PERIOD).mean()

    # VWAP approximation (daily)
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # Bollinger Bands
    df["BB_Mid"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Price change
    df["Pct_Change"] = df["Close"].pct_change() * 100

    return df


def detect_consolidation_breakout(df):
    """Detect if stock is breaking out of a consolidation range."""
    if len(df) < CONSOLIDATION_DAYS + 5:
        return None

    recent = df.tail(CONSOLIDATION_DAYS + 1)
    consol_zone = recent.head(CONSOLIDATION_DAYS)
    latest = recent.iloc[-1]

    high = consol_zone["High"].max()
    low = consol_zone["Low"].min()
    range_pct = ((high - low) / low) * 100

    if range_pct <= CONSOLIDATION_RANGE_PCT:
        if latest["Close"] > high and latest["Vol_Ratio"] >= BREAKOUT_VOLUME_MULTIPLIER:
            return {
                "pattern": "Consolidation Breakout",
                "breakout_level": round(high, 2),
                "range_low": round(low, 2),
                "range_high": round(high, 2),
                "consolidation_days": CONSOLIDATION_DAYS,
                "volume_confirmation": True,
            }
    return None


def detect_flag_breakout(df):
    """Detect bull flag pattern breakout."""
    if len(df) < 30:
        return None

    # Look for strong move (pole) followed by consolidation (flag)
    pole_zone = df.iloc[-30:-15]
    flag_zone = df.iloc[-15:-1]
    latest = df.iloc[-1]

    pole_move = ((pole_zone["Close"].iloc[-1] - pole_zone["Close"].iloc[0])
                 / pole_zone["Close"].iloc[0]) * 100

    if pole_move < 8:  # Need at least 8% pole
        return None

    flag_high = flag_zone["High"].max()
    flag_low = flag_zone["Low"].min()
    flag_range = ((flag_high - flag_low) / flag_low) * 100

    if flag_range <= 6 and latest["Close"] > flag_high:
        if latest["Vol_Ratio"] >= BREAKOUT_VOLUME_MULTIPLIER:
            return {
                "pattern": "Flag Breakout",
                "breakout_level": round(flag_high, 2),
                "pole_gain": round(pole_move, 2),
                "flag_range": round(flag_range, 2),
                "volume_confirmation": True,
            }
    return None


def detect_cup_and_handle(df):
    """Detect cup and handle pattern."""
    if len(df) < 40:
        return None

    cup_zone = df.iloc[-40:-10]
    handle_zone = df.iloc[-10:-1]
    latest = df.iloc[-1]

    cup_high = cup_zone["High"].max()
    cup_low = cup_zone["Low"].min()
    cup_depth = ((cup_high - cup_low) / cup_high) * 100

    if cup_depth < 10 or cup_depth > 35:
        return None

    # Cup should form a rounded bottom
    mid_point = len(cup_zone) // 2
    left_half_avg = cup_zone.iloc[:mid_point]["Close"].mean()
    right_half_avg = cup_zone.iloc[mid_point:]["Close"].mean()
    bottom_avg = cup_zone.iloc[mid_point-3:mid_point+3]["Close"].mean()

    if bottom_avg < left_half_avg and bottom_avg < right_half_avg:
        handle_high = handle_zone["High"].max()
        handle_retrace = ((cup_high - handle_zone["Low"].min()) / cup_high) * 100

        if handle_retrace < cup_depth * 0.5:
            if latest["Close"] > handle_high:
                return {
                    "pattern": "Cup and Handle",
                    "breakout_level": round(handle_high, 2),
                    "cup_depth": round(cup_depth, 2),
                    "handle_retrace": round(handle_retrace, 2),
                    "volume_confirmation": latest["Vol_Ratio"] >= BREAKOUT_VOLUME_MULTIPLIER,
                }
    return None


def detect_range_breakout(df):
    """Detect breakout from a well-defined trading range."""
    if len(df) < 25:
        return None

    range_zone = df.iloc[-25:-1]
    latest = df.iloc[-1]

    resistance = range_zone["High"].nlargest(3).mean()
    support = range_zone["Low"].nsmallest(3).mean()

    range_pct = ((resistance - support) / support) * 100

    if 3 <= range_pct <= 15:
        if latest["Close"] > resistance:
            if latest["Vol_Ratio"] >= BREAKOUT_VOLUME_MULTIPLIER:
                return {
                    "pattern": "Range Breakout",
                    "breakout_level": round(resistance, 2),
                    "support": round(support, 2),
                    "resistance": round(resistance, 2),
                    "range_pct": round(range_pct, 2),
                    "volume_confirmation": True,
                }
    return None


def detect_volume_breakout(df):
    """Detect significant volume spike with price action."""
    if len(df) < 5:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    if (latest["Vol_Ratio"] >= 2.0 and
            latest["Close"] > prev["Close"] and
            latest["Close"] > latest["Open"]):

        price_change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

        if 1.5 <= price_change <= 8:
            return {
                "pattern": "Volume Breakout",
                "breakout_level": round(prev["High"], 2),
                "volume_ratio": round(latest["Vol_Ratio"], 2),
                "price_change": round(price_change, 2),
                "volume_confirmation": True,
            }
    return None


def detect_ma_breakout(df):
    """Detect moving average crossover breakout."""
    if len(df) < 5:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Price crossing above 50 DMA with volume
    if (prev["Close"] < prev["SMA_50"] and
            latest["Close"] > latest["SMA_50"] and
            latest["Vol_Ratio"] >= 1.3):
        return {
            "pattern": "Moving Average Breakout (50 DMA)",
            "breakout_level": round(latest["SMA_50"], 2),
            "volume_confirmation": latest["Vol_Ratio"] >= BREAKOUT_VOLUME_MULTIPLIER,
        }

    # Price crossing above 200 DMA (golden cross potential)
    if (prev["Close"] < prev["SMA_200"] and
            latest["Close"] > latest["SMA_200"] and
            latest["Vol_Ratio"] >= 1.3):
        return {
            "pattern": "Moving Average Breakout (200 DMA)",
            "breakout_level": round(latest["SMA_200"], 2),
            "volume_confirmation": latest["Vol_Ratio"] >= BREAKOUT_VOLUME_MULTIPLIER,
        }
    return None


def detect_all_patterns(df):
    """Run all pattern detection algorithms. Return list of detected patterns."""
    patterns = []

    detectors = [
        detect_consolidation_breakout,
        detect_flag_breakout,
        detect_cup_and_handle,
        detect_range_breakout,
        detect_volume_breakout,
        detect_ma_breakout,
    ]

    for detector in detectors:
        result = detector(df)
        if result:
            patterns.append(result)

    return patterns


def compute_support_resistance(df, n_levels=3):
    """Compute key support and resistance levels using pivot points."""
    recent = df.tail(30)

    # Resistance: recent highs
    resistances = sorted(recent["High"].nlargest(n_levels * 2).unique(), reverse=True)[:n_levels]

    # Support: recent lows
    supports = sorted(recent["Low"].nsmallest(n_levels * 2).unique())[:n_levels]

    return {
        "resistance": [round(r, 2) for r in resistances],
        "support": [round(s, 2) for s in supports],
    }


def calculate_trade_levels(df, breakout_pattern):
    """Calculate entry, stop loss, and target levels."""
    latest = df.iloc[-1]
    close = latest["Close"]
    atr = latest["ATR"] if not pd.isna(latest["ATR"]) else close * 0.02

    breakout_level = breakout_pattern.get("breakout_level", close)

    # Entry zone: around breakout level
    entry_low = round(breakout_level * 0.995, 2)
    entry_high = round(breakout_level * 1.01, 2)

    # Stop loss: below breakout level or ATR-based
    sl_atr = round(breakout_level - (1.5 * atr), 2)
    sl_pattern = breakout_pattern.get("range_low",
                 breakout_pattern.get("support", sl_atr))
    stop_loss = round(min(sl_atr, sl_pattern) if isinstance(sl_pattern, (int, float))
                      else sl_atr, 2)

    # Targets
    target_1 = round(close * (1 + SWING_TARGET_1_PCT / 100), 2)
    target_2 = round(close * (1 + SWING_TARGET_2_PCT / 100), 2)
    target_3 = round(close * (1 + SWING_TARGET_3_PCT / 100), 2)

    # Risk-Reward
    risk = close - stop_loss
    reward = target_2 - close
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    return {
        "entry_zone": f"₹{entry_low} – ₹{entry_high}",
        "entry_low": entry_low,
        "entry_high": entry_high,
        "breakout_confirmation": f"₹{breakout_level}",
        "stop_loss": f"₹{stop_loss}",
        "stop_loss_val": stop_loss,
        "target_1": f"₹{target_1} (+{SWING_TARGET_1_PCT}%)",
        "target_2": f"₹{target_2} (+{SWING_TARGET_2_PCT}%)",
        "target_3": f"₹{target_3} (+{SWING_TARGET_3_PCT}%)",
        "target_1_val": target_1,
        "target_2_val": target_2,
        "target_3_val": target_3,
        "risk_reward": rr_ratio,
        "risk_amount": round(risk, 2),
        "reward_amount": round(reward, 2),
    }


def compute_probability_score(df, patterns, trade_levels):
    """Compute a probability score (0-10) for the trade setup."""
    score = 0
    latest = df.iloc[-1]

    # 1. Price above 50 DMA (+1)
    if latest["Close"] > latest["SMA_50"]:
        score += 1

    # 2. Price above 200 DMA (+1)
    if latest["Close"] > latest["SMA_200"]:
        score += 1

    # 3. RSI in sweet spot 45-70 (+1)
    rsi = latest["RSI"]
    if 45 <= rsi <= 70:
        score += 1
    elif 35 <= rsi < 45:
        score += 0.5

    # 4. Volume confirmation (+1.5)
    if latest["Vol_Ratio"] >= BREAKOUT_VOLUME_MULTIPLIER:
        score += 1.5
    elif latest["Vol_Ratio"] >= 1.2:
        score += 0.5

    # 5. Number of patterns detected (+1 per pattern, max 2)
    score += min(len(patterns), 2)

    # 6. Risk-reward ratio (+1 if >= 2)
    if trade_levels["risk_reward"] >= MIN_RISK_REWARD:
        score += 1

    # 7. MACD bullish (+0.5)
    if latest["MACD_Hist"] > 0:
        score += 0.5

    # 8. Price above EMA 20 (+0.5)
    if latest["Close"] > latest["EMA_20"]:
        score += 0.5

    return min(round(score, 1), 10)


def get_trend(df):
    """Determine the current trend of the stock."""
    latest = df.iloc[-1]

    above_50 = latest["Close"] > latest["SMA_50"]
    above_200 = latest["Close"] > latest["SMA_200"]
    sma50_above_200 = latest["SMA_50"] > latest["SMA_200"]

    if above_50 and above_200 and sma50_above_200:
        return "Strong Uptrend"
    elif above_50 and above_200:
        return "Uptrend"
    elif above_50:
        return "Moderate Bullish"
    elif not above_50 and not above_200:
        return "Downtrend"
    else:
        return "Sideways"


def get_rsi_analysis(rsi_value):
    """Return RSI analysis text."""
    if pd.isna(rsi_value):
        return "N/A"
    rsi = round(rsi_value, 1)
    if rsi >= 75:
        return f"{rsi} — Overbought (Caution)"
    elif rsi >= 60:
        return f"{rsi} — Bullish Momentum"
    elif rsi >= 45:
        return f"{rsi} — Neutral-Bullish"
    elif rsi >= 30:
        return f"{rsi} — Weakening"
    else:
        return f"{rsi} — Oversold (Potential Reversal)"


def get_volume_analysis(vol_ratio):
    """Return volume analysis text."""
    if pd.isna(vol_ratio):
        return "N/A"
    vr = round(vol_ratio, 2)
    if vr >= 3.0:
        return f"{vr}x avg — Exceptional Volume Surge"
    elif vr >= 2.0:
        return f"{vr}x avg — Strong Volume Breakout"
    elif vr >= 1.5:
        return f"{vr}x avg — Above Average Volume"
    elif vr >= 1.0:
        return f"{vr}x avg — Normal Volume"
    else:
        return f"{vr}x avg — Below Average (Weak)"
