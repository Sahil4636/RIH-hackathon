"""
whale_alert.py
--------------
Fetches large crypto transactions from the Whale Alert API.
Free tier: 10 requests/minute, last 3600 seconds of data.

Sign up for a free API key at: https://whale-alert.io/
"""

import requests
import pandas as pd
from datetime import datetime, timezone
import time

WHALE_BASE = "https://api.whale-alert.io/v1"

# Minimum USD value to consider a "whale" transaction
WHALE_MIN_USD = 500_000   # $500k+

# Blockchain symbols mapped to coin ids
CHAIN_TO_COIN = {
    "bitcoin":        "bitcoin",
    "ethereum":       "ethereum",
    "binance_chain":  "binancecoin",
    "solana":         "solana",
    "ripple":         "ripple",
    "cardano":        "cardano",
    "avalanche":      "avalanche-2",
    "polygon":        "matic-network",
    "tron":           "tron",
    "dogecoin":       "dogecoin",
}

TX_TYPE_RISK = {
    "transfer":       2,   # wallet-to-wallet, moderate
    "exchange_to_exchange": 1,
    "wallet_to_exchange":   4,   # selling pressure
    "exchange_to_wallet":   1,   # accumulation (low risk)
    "mint":           3,
    "burn":           1,
    "unknown":        2,
}


# ─────────────────────────────────────────────
# FETCH TRANSACTIONS
# ─────────────────────────────────────────────
def fetch_whale_transactions(api_key: str,
                              min_usd: int = WHALE_MIN_USD,
                              lookback_seconds: int = 3600,
                              limit: int = 100) -> pd.DataFrame:
    """
    Returns recent whale transactions above min_usd threshold.
    lookback_seconds: how far back to look (max 3600 on free tier)
    """
    if not api_key or api_key == "YOUR_API_KEY":
        return _demo_data()

    url = f"{WHALE_BASE}/transactions"
    params = {
        "api_key":  api_key,
        "min_value": min_usd,
        "limit":    limit,
        "start":    int(time.time()) - lookback_seconds,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        txs  = data.get("transactions", [])
        if not txs:
            return pd.DataFrame()
        return _parse_transactions(txs)
    except Exception as e:
        print(f"[Whale Alert] API error: {e}")
        return _demo_data()


def _parse_transactions(txs: list) -> pd.DataFrame:
    rows = []
    for tx in txs:
        amount_usd = float(tx.get("amount_usd", 0))
        blockchain  = tx.get("blockchain", "unknown").lower()
        symbol      = tx.get("symbol", "").upper()
        tx_type     = _classify_tx(tx)
        from_owner  = tx.get("from", {}).get("owner_type", "unknown")
        to_owner    = tx.get("to",   {}).get("owner_type", "unknown")

        rows.append({
            "timestamp":    datetime.fromtimestamp(
                                tx.get("timestamp", time.time()),
                                tz=timezone.utc
                            ),
            "blockchain":   blockchain,
            "symbol":       symbol,
            "coin_id":      CHAIN_TO_COIN.get(blockchain, blockchain),
            "amount":       float(tx.get("amount", 0)),
            "amount_usd":   amount_usd,
            "tx_type":      tx_type,
            "from_type":    from_owner,
            "to_type":      to_owner,
            "hash":         tx.get("hash", "")[:16] + "…",
            "risk_weight":  TX_TYPE_RISK.get(tx_type, 2),
        })
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False)


def _classify_tx(tx: dict) -> str:
    from_type = tx.get("from", {}).get("owner_type", "unknown")
    to_type   = tx.get("to",   {}).get("owner_type", "unknown")
    if from_type == "exchange" and to_type == "exchange":
        return "exchange_to_exchange"
    if from_type == "wallet"   and to_type == "exchange":
        return "wallet_to_exchange"
    if from_type == "exchange" and to_type == "wallet":
        return "exchange_to_wallet"
    return tx.get("transaction_type", "transfer")


# ─────────────────────────────────────────────
# DEMO DATA (shown when no API key provided)
# ─────────────────────────────────────────────
def _demo_data() -> pd.DataFrame:
    now = datetime.now(tz=timezone.utc)
    demo = [
        {"timestamp": now, "blockchain": "bitcoin", "symbol": "BTC",
         "coin_id": "bitcoin", "amount": 1200, "amount_usd": 72_000_000,
         "tx_type": "wallet_to_exchange", "from_type": "wallet",
         "to_type": "exchange", "hash": "3f7a1b2c9d…", "risk_weight": 4},
        {"timestamp": now, "blockchain": "ethereum", "symbol": "ETH",
         "coin_id": "ethereum", "amount": 18000, "amount_usd": 54_000_000,
         "tx_type": "exchange_to_wallet", "from_type": "exchange",
         "to_type": "wallet", "hash": "9e2d8f4a1c…", "risk_weight": 1},
        {"timestamp": now, "blockchain": "ripple", "symbol": "XRP",
         "coin_id": "ripple", "amount": 50_000_000, "amount_usd": 25_000_000,
         "tx_type": "transfer", "from_type": "unknown",
         "to_type": "unknown", "hash": "7b3c5e9a2f…", "risk_weight": 2},
        {"timestamp": now, "blockchain": "ethereum", "symbol": "USDT",
         "coin_id": "ethereum", "amount": 30_000_000, "amount_usd": 30_000_000,
         "tx_type": "wallet_to_exchange", "from_type": "wallet",
         "to_type": "exchange", "hash": "1a4d6f8b3e…", "risk_weight": 4},
        {"timestamp": now, "blockchain": "solana", "symbol": "SOL",
         "coin_id": "solana", "amount": 400_000, "amount_usd": 60_000_000,
         "tx_type": "wallet_to_exchange", "from_type": "wallet",
         "to_type": "exchange", "hash": "5c8e2a7d1f…", "risk_weight": 4},
        {"timestamp": now, "blockchain": "bitcoin", "symbol": "BTC",
         "coin_id": "bitcoin", "amount": 500, "amount_usd": 30_000_000,
         "tx_type": "exchange_to_wallet", "from_type": "exchange",
         "to_type": "wallet", "hash": "2d9b4f6a8c…", "risk_weight": 1},
    ]
    return pd.DataFrame(demo)


# ─────────────────────────────────────────────
# PER-COIN WHALE RISK SIGNALS
# ─────────────────────────────────────────────
def compute_whale_signals(whale_df: pd.DataFrame,
                           coin_ids: list) -> pd.DataFrame:
    """
    Aggregates whale transactions per coin into 3 risk signals:
      - whale_tx_count      : number of whale txs in last hour
      - whale_volume_usd    : total USD moved by whales
      - whale_risk_score    : weighted risk score (0–100)
    """
    results = []
    for cid in coin_ids:
        subset = whale_df[whale_df["coin_id"] == cid] if not whale_df.empty else pd.DataFrame()

        tx_count   = len(subset)
        vol_usd    = float(subset["amount_usd"].sum()) if not subset.empty else 0.0
        risk_score = 0.0

        if not subset.empty:
            # Weight by risk_weight and USD amount
            weighted = (subset["risk_weight"] * subset["amount_usd"]).sum()
            max_possible = 4 * vol_usd if vol_usd > 0 else 1
            risk_score = min((weighted / max_possible) * 100, 100)

            # Boost score for high volumes (>$100M in an hour = high risk)
            vol_boost = min(vol_usd / 100_000_000 * 30, 30)
            risk_score = min(risk_score + vol_boost, 100)

        results.append({
            "coin_id":          cid,
            "whale_tx_count":   tx_count,
            "whale_volume_usd": round(vol_usd, 2),
            "whale_risk_score": round(risk_score, 2),
        })

    return pd.DataFrame(results).set_index("coin_id")


# ─────────────────────────────────────────────
# ALERT LEVEL
# ─────────────────────────────────────────────
def whale_alert_level(risk_score: float) -> tuple:
    if risk_score >= 70:
        return "🔴 HIGH ALERT",  "#e74c3c"
    if risk_score >= 40:
        return "🟠 MODERATE",    "#e67e22"
    if risk_score >= 10:
        return "🟡 LOW",         "#f1c40f"
    return     "🟢 QUIET",       "#2ecc71"


if __name__ == "__main__":
    print("=== Demo Whale Transactions ===")
    df = fetch_whale_transactions("YOUR_API_KEY")
    print(df[["symbol", "amount_usd", "tx_type", "risk_weight"]].to_string())

    print("\n=== Whale Signals per Coin ===")
    signals = compute_whale_signals(df, ["bitcoin", "ethereum", "solana", "ripple"])
    print(signals.to_string())
