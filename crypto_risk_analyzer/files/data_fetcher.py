"""
data_fetcher.py
---------------
Fetches crypto data from:
  - CoinGecko API  (prices, market data, historical OHLCV)
  - Alternative.me (Fear & Greed Index)

No API keys required for the free tiers used here.
"""

import requests
import pandas as pd
import time
from datetime import datetime


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COINGECKO_BASE  = "https://api.coingecko.com/api/v3"
FEAR_GREED_URL  = "https://api.alternative.me/fng/"

TOP_COINS = [
    "bitcoin", "ethereum", "binancecoin", "solana", "ripple",
    "cardano", "dogecoin", "avalanche-2", "polkadot", "chainlink"
]

HEADERS = {"accept": "application/json"}


# ─────────────────────────────────────────────
# 1. MARKET OVERVIEW (prices + 24h stats)
# ─────────────────────────────────────────────
def fetch_market_overview(coin_ids: list = TOP_COINS) -> pd.DataFrame:
    """
    Returns a DataFrame with current price, market cap, volume,
    24h price change (%), and 7d price change (%) for each coin.
    """
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ",".join(coin_ids),
        "order": "market_cap_desc",
        "sparkline": False,
        "price_change_percentage": "24h,7d",
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for c in data:
        rows.append({
            "id":               c["id"],
            "symbol":           c["symbol"].upper(),
            "name":             c["name"],
            "price_usd":        c["current_price"],
            "market_cap":       c["market_cap"],
            "volume_24h":       c["total_volume"],
            "change_24h_pct":   c.get("price_change_percentage_24h", 0),
            "change_7d_pct":    c.get("price_change_percentage_7d_in_currency", 0),
            "ath":              c["ath"],
            "ath_change_pct":   c["ath_change_percentage"],   # how far from ATH
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 2. HISTORICAL OHLCV (for volatility calc)
# ─────────────────────────────────────────────
def fetch_historical_prices(coin_id: str, days: int = 30) -> pd.DataFrame:
    """
    Returns daily closing prices for the last `days` days.
    CoinGecko returns OHLCV at daily granularity when days >= 90,
    and at hourly granularity for days < 90 — we resample to daily.
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    prices = raw.get("prices", [])
    volumes = raw.get("total_volumes", [])

    df_price  = pd.DataFrame(prices,  columns=["timestamp_ms", "close"])
    df_volume = pd.DataFrame(volumes, columns=["timestamp_ms", "volume"])

    df = df_price.merge(df_volume, on="timestamp_ms")
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.normalize()
    df = df.drop_duplicates("date").set_index("date").drop(columns="timestamp_ms")
    df["coin"] = coin_id
    return df


# ─────────────────────────────────────────────
# 3. FEAR & GREED INDEX
# ─────────────────────────────────────────────
def fetch_fear_greed(limit: int = 30) -> pd.DataFrame:
    """
    Returns the last `limit` days of Fear & Greed values (0–100).
    0  = Extreme Fear  |  100 = Extreme Greed
    """
    params = {"limit": limit, "format": "json"}
    resp = requests.get(FEAR_GREED_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    rows = [{
        "date":        datetime.utcfromtimestamp(int(d["timestamp"])).date(),
        "fg_value":    int(d["value"]),
        "fg_label":    d["value_classification"],
    } for d in data]

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. COIN DETAIL (description + links)
# ─────────────────────────────────────────────
def fetch_coin_detail(coin_id: str) -> dict:
    """
    Returns basic metadata: description, homepage, genesis date, etc.
    (Rate-limited — call sparingly.)
    """
    url = f"{COINGECKO_BASE}/coins/{coin_id}"
    params = {
        "localization": False,
        "tickers": False,
        "market_data": False,
        "community_data": False,
        "developer_data": False,
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    return {
        "id":           raw["id"],
        "name":         raw["name"],
        "symbol":       raw["symbol"].upper(),
        "description":  raw["description"].get("en", "")[:500],
        "homepage":     (raw["links"]["homepage"] or [""])[0],
        "genesis_date": raw.get("genesis_date"),
        "categories":   raw.get("categories", []),
    }


# ─────────────────────────────────────────────
# 5. BULK HISTORICAL (all top coins)
# ─────────────────────────────────────────────
def fetch_all_historical(coin_ids: list = TOP_COINS,
                          days: int = 30,
                          sleep_sec: float = 1.2) -> pd.DataFrame:
    """
    Fetches historical data for all coins with a small delay
    to respect CoinGecko's free-tier rate limit (~10–30 req/min).
    """
    frames = []
    for cid in coin_ids:
        try:
            df = fetch_historical_prices(cid, days=days)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Could not fetch {cid}: {e}")
        time.sleep(sleep_sec)
    return pd.concat(frames) if frames else pd.DataFrame()


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Market Overview ===")
    market = fetch_market_overview(["bitcoin", "ethereum", "solana"])
    print(market[["symbol", "price_usd", "change_24h_pct", "change_7d_pct"]].to_string())

    print("\n=== Fear & Greed (last 7 days) ===")
    fg = fetch_fear_greed(limit=7)
    print(fg.to_string())

    print("\n=== BTC Historical (7 days) ===")
    hist = fetch_historical_prices("bitcoin", days=7)
    print(hist.to_string())
