"""
Data Fetcher — Downloads historical OHLCV data from Yahoo Finance for NSE stocks.
Uses caching to avoid redundant API calls within the same session.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

from config import LOOKBACK_DAYS
from nse_stocks import get_all_tickers, INDEX_TICKERS

logger = logging.getLogger(__name__)

_cache = {}
_cache_timestamp = {}
CACHE_TTL = 300  # 5 minutes


def fetch_stock_data(ticker, period_days=LOOKBACK_DAYS):
    """Fetch OHLCV data for a single NSE stock."""
    cache_key = f"{ticker}_{period_days}"
    now = time.time()

    if cache_key in _cache and (now - _cache_timestamp.get(cache_key, 0)) < CACHE_TTL:
        return _cache[cache_key]

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 50)  # extra buffer

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for {ticker}")
            return None

        df = df.tail(period_days)
        _cache[cache_key] = df
        _cache_timestamp[cache_key] = now
        return df

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return None


def fetch_multiple_stocks(tickers=None, period_days=LOOKBACK_DAYS):
    """Fetch data for multiple stocks. Returns dict of {ticker: DataFrame}."""
    if tickers is None:
        tickers = get_all_tickers()

    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        logger.info(f"Fetching {ticker} ({i+1}/{total})")
        df = fetch_stock_data(ticker, period_days)
        if df is not None:
            results[ticker] = df

    logger.info(f"Fetched data for {len(results)}/{total} stocks")
    return results


def fetch_index_data(period_days=LOOKBACK_DAYS):
    """Fetch data for Nifty, Bank Nifty, and sector indices."""
    results = {}
    for name, ticker in INDEX_TICKERS.items():
        df = fetch_stock_data(ticker, period_days)
        if df is not None:
            results[name] = df
    return results


def get_stock_info(ticker):
    """Get basic stock info (market cap, sector, etc.)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker.replace(".NS", "")),
            "market_cap": info.get("marketCap", 0),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "avg_volume": info.get("averageVolume", 0),
        }
    except Exception:
        return {"name": ticker.replace(".NS", ""), "market_cap": 0,
                "sector": "Unknown", "industry": "Unknown", "avg_volume": 0}


def clear_cache():
    """Clear the data cache."""
    _cache.clear()
    _cache_timestamp.clear()
