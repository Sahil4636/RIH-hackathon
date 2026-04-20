"""
backend/api.py
--------------
API route functions — called by the Streamlit frontend.
Acts as a clean interface layer between UI and business logic.
"""

from backend.data_fetcher import (
    fetch_market_overview, fetch_historical_prices,
    fetch_all_historical, fetch_fear_greed,
    fetch_coin_news_headlines, TOP_COINS,
)
from backend.risk_engine import compute_full_risk, compute_all_risks


def get_risk_for_coin(coin_id: str, days: int = 90) -> dict:
    """
    Single-coin risk assessment.
    Returns full API response dict.
    """
    try:
        market_df  = fetch_market_overview([coin_id])
        hist_df    = fetch_historical_prices(coin_id, days=days)
        fg_df      = fetch_fear_greed(limit=30)
        headlines  = fetch_coin_news_headlines(coin_id)

        if market_df.empty:
            return _error_response(coin_id, "Market data unavailable")

        market_row = market_df.iloc[0]
        return compute_full_risk(coin_id, market_row, hist_df, fg_df, headlines)

    except Exception as e:
        return _error_response(coin_id, str(e))


def get_risk_for_all(coin_ids: list = TOP_COINS, days: int = 90) -> list[dict]:
    """
    Batch risk assessment for multiple coins.
    Returns list of risk response dicts sorted by risk_score desc.
    """
    try:
        market_df = fetch_market_overview(coin_ids)
        hist_df   = fetch_all_historical(coin_ids, days=days)
        fg_df     = fetch_fear_greed(limit=30)
        headlines_map = {cid: fetch_coin_news_headlines(cid) for cid in coin_ids[:3]}
        return compute_all_risks(market_df, hist_df, fg_df, headlines_map)
    except Exception as e:
        return [_error_response(cid, str(e)) for cid in coin_ids]


def get_market_overview(coin_ids: list = TOP_COINS):
    """Returns raw market dataframe."""
    return fetch_market_overview(coin_ids)


def get_historical(coin_id: str, days: int = 90):
    """Returns historical OHLCV dataframe for one coin."""
    return fetch_historical_prices(coin_id, days=days)


def get_fear_greed(limit: int = 30):
    """Returns Fear & Greed dataframe."""
    return fetch_fear_greed(limit=limit)


def _error_response(coin_id: str, message: str) -> dict:
    return {
        "coin":       coin_id,
        "symbol":     coin_id.upper(),
        "name":       coin_id.title(),
        "price_usd":  0,
        "risk_score": 50,
        "risk_level": "Unknown",
        "risk_icon":  "❓",
        "risk_color": "#888888",
        "reason":     [f"⚠️ Data unavailable: {message}"],
        "suggestion": "⏸️ Hold — Unable to assess risk accurately.",
        "details":    {},
        "error":      True,
    }
