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
    """Fetch data for multiple stocks using batch download for speed."""
    if tickers is None:
        tickers = get_all_tickers()

    results = {}
    total = len(tickers)

    # Check cache first, collect uncached tickers
    now = time.time()
    uncached = []
    for ticker in tickers:
        cache_key = f"{ticker}_{period_days}"
        if cache_key in _cache and (now - _cache_timestamp.get(cache_key, 0)) < CACHE_TTL:
            results[ticker] = _cache[cache_key]
        else:
            uncached.append(ticker)

    if uncached:
        logger.info(f"Batch downloading {len(uncached)} stocks ({len(results)} from cache)...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 50)

        try:
            # yf.download with multiple tickers is much faster (single network call)
            batch_df = yf.download(
                uncached,
                start=start_date,
                end=end_date,
                group_by="ticker",
                threads=True,
                progress=False,
            )

            if batch_df is not None and not batch_df.empty:
                for ticker in uncached:
                    try:
                        if len(uncached) == 1:
                            df = batch_df.copy()
                        else:
                            df = batch_df[ticker].copy()

                        df = df.dropna(how="all")
                        if len(df) < 30:
                            logger.warning(f"Insufficient data for {ticker}")
                            continue

                        df = df.tail(period_days)
                        cache_key = f"{ticker}_{period_days}"
                        _cache[cache_key] = df
                        _cache_timestamp[cache_key] = now
                        results[ticker] = df
                    except (KeyError, Exception) as e:
                        logger.warning(f"Failed to extract {ticker} from batch: {e}")
        except Exception as e:
            logger.error(f"Batch download failed: {e}, falling back to sequential")
            for ticker in uncached:
                df = fetch_stock_data(ticker, period_days)
                if df is not None:
                    results[ticker] = df

    logger.info(f"Fetched data for {len(results)}/{total} stocks")
    return results


def fetch_index_data(period_days=LOOKBACK_DAYS):
    """Fetch data for Nifty, Bank Nifty, and sector indices (batch)."""
    results = {}
    index_tickers = list(INDEX_TICKERS.values())

    now = time.time()
    uncached_names = []
    uncached_tickers = []
    for name, ticker in INDEX_TICKERS.items():
        cache_key = f"{ticker}_{period_days}"
        if cache_key in _cache and (now - _cache_timestamp.get(cache_key, 0)) < CACHE_TTL:
            results[name] = _cache[cache_key]
        else:
            uncached_names.append(name)
            uncached_tickers.append(ticker)

    if uncached_tickers:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 50)
        try:
            batch_df = yf.download(
                uncached_tickers,
                start=start_date,
                end=end_date,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if batch_df is not None and not batch_df.empty:
                for name, ticker in zip(uncached_names, uncached_tickers):
                    try:
                        if len(uncached_tickers) == 1:
                            df = batch_df.copy()
                        else:
                            df = batch_df[ticker].copy()
                        df = df.dropna(how="all")
                        if len(df) >= 30:
                            df = df.tail(period_days)
                            cache_key = f"{ticker}_{period_days}"
                            _cache[cache_key] = df
                            _cache_timestamp[cache_key] = now
                            results[name] = df
                    except (KeyError, Exception):
                        pass
        except Exception:
            for name, ticker in zip(uncached_names, uncached_tickers):
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
