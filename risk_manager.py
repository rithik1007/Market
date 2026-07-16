"""
Risk Manager — Portfolio-level risk analysis.
Correlation, sector concentration, position sizing, VaR, drawdown tracking.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from data_fetcher import fetch_stock_data
from nse_stocks import get_stock_sector

logger = logging.getLogger(__name__)


def analyze_portfolio_risk(trade_plans: list, capital: int = 100000) -> dict:
    """
    Analyze risk of proposed trade plans.
    Returns sector concentration, correlation matrix, position sizing,
    portfolio VaR, and risk warnings.
    """
    if not trade_plans:
        return {"risk_score": 0, "warnings": [], "analysis": {}}

    tickers = []
    allocations = []
    sectors = []

    for plan in trade_plans:
        ticker = plan.get("ticker", "")
        if not ticker:
            continue
        tickers.append(ticker)
        cap_str = str(plan.get("capital_to_invest", "0"))
        alloc = _parse_amount(cap_str)
        allocations.append(alloc if alloc else 0)
        sectors.append(get_stock_sector(ticker + ".NS"))

    if not tickers:
        return {"risk_score": 0, "warnings": [], "analysis": {}}

    total_deployed = sum(allocations)
    cash_pct = ((capital - total_deployed) / capital * 100) if capital > 0 else 0

    # 1. Sector Concentration
    sector_conc = _compute_sector_concentration(tickers, allocations, sectors)

    # 2. Correlation Matrix
    correlation = _compute_correlation(tickers)

    # 3. Position Sizing Analysis
    sizing = _analyze_position_sizing(tickers, allocations, capital)

    # 4. Portfolio VaR (95% confidence, 1-day)
    var_data = _compute_var(tickers, allocations)

    # 5. Risk Warnings
    warnings = _generate_risk_warnings(
        sector_conc, correlation, sizing, var_data, cash_pct, len(tickers)
    )

    # Overall risk score (0-100, higher = riskier)
    risk_score = _compute_risk_score(sector_conc, correlation, sizing, var_data, cash_pct)

    return {
        "risk_score": round(risk_score, 0),
        "risk_level": "HIGH" if risk_score >= 70 else ("MODERATE" if risk_score >= 40 else "LOW"),
        "total_deployed": round(total_deployed, 0),
        "cash_pct": round(cash_pct, 1),
        "num_positions": len(tickers),
        "sector_concentration": sector_conc,
        "correlation_matrix": correlation,
        "position_sizing": sizing,
        "var_95": var_data,
        "warnings": warnings,
    }


def _compute_sector_concentration(tickers, allocations, sectors):
    """Compute sector concentration (Herfindahl index)."""
    total = sum(allocations) if sum(allocations) > 0 else 1
    sector_alloc = {}
    for i, sector in enumerate(sectors):
        if sector not in sector_alloc:
            sector_alloc[sector] = 0
        sector_alloc[sector] += allocations[i]

    sector_pcts = {}
    hhi = 0
    for sector, alloc in sector_alloc.items():
        pct = (alloc / total) * 100
        sector_pcts[sector] = round(pct, 1)
        hhi += (pct / 100) ** 2

    # Max single sector exposure
    max_sector = max(sector_pcts.values()) if sector_pcts else 0

    return {
        "sectors": sector_pcts,
        "hhi": round(hhi, 3),  # 1.0 = fully concentrated, <0.25 = diversified
        "max_sector_exposure": round(max_sector, 1),
        "num_sectors": len(sector_pcts),
        "diversified": hhi < 0.4,
    }


def _compute_correlation(tickers):
    """Compute pairwise correlation between stocks."""
    if len(tickers) < 2:
        return {"avg_correlation": 0, "high_pairs": [], "matrix": {}}

    price_data = {}
    for ticker in tickers:
        try:
            df = fetch_stock_data(ticker + ".NS", period_days=60)
            if df is not None and len(df) >= 20:
                price_data[ticker] = df["Close"].pct_change().dropna().values
        except Exception:
            continue

    if len(price_data) < 2:
        return {"avg_correlation": 0, "high_pairs": [], "matrix": {}}

    # Align lengths
    min_len = min(len(v) for v in price_data.values())
    aligned = {k: v[-min_len:] for k, v in price_data.items()}

    keys = list(aligned.keys())
    n = len(keys)
    correlations = []
    high_pairs = []
    matrix = {}

    for i in range(n):
        matrix[keys[i]] = {}
        for j in range(n):
            if i == j:
                matrix[keys[i]][keys[j]] = 1.0
                continue
            corr = float(np.corrcoef(aligned[keys[i]], aligned[keys[j]])[0, 1])
            matrix[keys[i]][keys[j]] = round(corr, 2)
            if i < j:
                correlations.append(corr)
                if abs(corr) > 0.7:
                    high_pairs.append({
                        "pair": f"{keys[i]}-{keys[j]}",
                        "correlation": round(corr, 2),
                    })

    avg_corr = np.mean(correlations) if correlations else 0

    return {
        "avg_correlation": round(float(avg_corr), 2),
        "high_pairs": high_pairs,
        "matrix": matrix,
    }


def _analyze_position_sizing(tickers, allocations, capital):
    """Analyze position sizing relative to capital."""
    total = sum(allocations) if sum(allocations) > 0 else 1
    positions = []
    max_position_pct = 0
    for i, ticker in enumerate(tickers):
        pct = (allocations[i] / capital * 100) if capital > 0 else 0
        positions.append({
            "ticker": ticker,
            "allocation": round(allocations[i], 0),
            "pct_of_capital": round(pct, 1),
        })
        max_position_pct = max(max_position_pct, pct)

    return {
        "positions": positions,
        "max_position_pct": round(max_position_pct, 1),
        "avg_position_pct": round((sum(a for a in allocations) / len(allocations) / capital * 100) if allocations and capital else 0, 1),
        "concentrated": max_position_pct > 40,
    }


def _compute_var(tickers, allocations):
    """Compute portfolio Value at Risk (95% confidence, 1-day)."""
    total_alloc = sum(allocations)
    if total_alloc == 0:
        return {"var_amount": 0, "var_pct": 0}

    weighted_returns = []
    for i, ticker in enumerate(tickers):
        weight = allocations[i] / total_alloc if total_alloc > 0 else 0
        try:
            df = fetch_stock_data(ticker + ".NS", period_days=60)
            if df is not None and len(df) >= 20:
                returns = df["Close"].pct_change().dropna().values
                weighted_returns.append(returns * weight)
        except Exception:
            continue

    if not weighted_returns:
        return {"var_amount": 0, "var_pct": 0}

    min_len = min(len(r) for r in weighted_returns)
    portfolio_returns = sum(r[-min_len:] for r in weighted_returns)

    var_95 = float(np.percentile(portfolio_returns, 5))  # 5th percentile = 95% VaR
    var_amount = abs(var_95 * total_alloc)

    return {
        "var_amount": round(var_amount, 0),
        "var_pct": round(abs(var_95) * 100, 2),
        "worst_day": round(float(np.min(portfolio_returns)) * 100, 2),
        "best_day": round(float(np.max(portfolio_returns)) * 100, 2),
    }


def _generate_risk_warnings(sector_conc, correlation, sizing, var_data, cash_pct, num_positions):
    """Generate risk warnings based on analysis."""
    warnings = []

    if sector_conc["max_sector_exposure"] > 50:
        top_sector = max(sector_conc["sectors"], key=sector_conc["sectors"].get)
        warnings.append({
            "level": "HIGH",
            "message": f"Heavy sector concentration: {sector_conc['max_sector_exposure']}% in {top_sector}",
        })

    if correlation.get("avg_correlation", 0) > 0.6:
        warnings.append({
            "level": "HIGH",
            "message": f"High portfolio correlation ({correlation['avg_correlation']}). Stocks move together — limited diversification.",
        })

    for pair in correlation.get("high_pairs", [])[:3]:
        warnings.append({
            "level": "MEDIUM",
            "message": f"Correlated pair: {pair['pair']} (r={pair['correlation']})",
        })

    if sizing.get("max_position_pct", 0) > 50:
        warnings.append({
            "level": "HIGH",
            "message": f"Single position is {sizing['max_position_pct']}% of capital — high concentration risk.",
        })

    if cash_pct < 10:
        warnings.append({
            "level": "MEDIUM",
            "message": f"Only {cash_pct}% cash remaining — no buffer for dips.",
        })

    if num_positions == 1:
        warnings.append({
            "level": "MEDIUM",
            "message": "Single stock portfolio — no diversification.",
        })

    var_pct = var_data.get("var_pct", 0)
    if var_pct > 3:
        warnings.append({
            "level": "HIGH",
            "message": f"Daily VaR is {var_pct}% — high daily risk exposure.",
        })

    return warnings


def _compute_risk_score(sector_conc, correlation, sizing, var_data, cash_pct):
    """Compute overall portfolio risk score (0-100)."""
    score = 0

    # Sector concentration (0-25)
    hhi = sector_conc.get("hhi", 0)
    score += min(25, hhi * 25)

    # Correlation (0-25)
    avg_corr = abs(correlation.get("avg_correlation", 0))
    score += min(25, avg_corr * 25)

    # Position concentration (0-25)
    max_pos = sizing.get("max_position_pct", 0)
    score += min(25, max_pos * 0.5)

    # Cash buffer (0-25)
    if cash_pct < 5:
        score += 25
    elif cash_pct < 20:
        score += 15
    elif cash_pct < 40:
        score += 5

    return min(100, score)


def _parse_amount(amount_str) -> Optional[float]:
    """Parse amount string like '₹50,000' to float."""
    try:
        cleaned = str(amount_str).replace("₹", "").replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None
