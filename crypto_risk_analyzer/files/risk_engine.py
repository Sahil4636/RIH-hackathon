"""
risk_engine.py
--------------
Computes a composite Risk Score (0–100) for each coin using:
  - Volatility (30-day rolling std of daily returns)
  - Liquidity ratio (volume / market cap)
  - Drawdown from ATH
  - 24-hour price momentum
  - Fear & Greed Index (market-wide sentiment)
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# WEIGHTS (must sum to 1.0)
# ─────────────────────────────────────────────
WEIGHTS = {
    "volatility":   0.30,
    "drawdown":     0.25,
    "momentum":     0.20,
    "liquidity":    0.15,
    "fear_greed":   0.10,
}

RISK_LABELS = {
    (0,  25):  ("🟢 Low",      "#2ecc71"),
    (25, 50):  ("🟡 Medium",   "#f1c40f"),
    (50, 75):  ("🟠 High",     "#e67e22"),
    (75, 101): ("🔴 Extreme",  "#e74c3c"),
}


def _label(score: float):
    for (lo, hi), (label, color) in RISK_LABELS.items():
        if lo <= score < hi:
            return label, color
    return "🔴 Extreme", "#e74c3c"


# ─────────────────────────────────────────────
# INDIVIDUAL COMPONENTS (each returns 0–100)
# ─────────────────────────────────────────────

def compute_volatility_score(hist_df: pd.DataFrame) -> pd.Series:
    """
    For each coin: 30-day rolling std of daily % returns → normalised to 0–100.
    Higher std  →  higher risk score.
    """
    results = {}
    for coin, grp in hist_df.groupby("coin"):
        grp = grp.sort_index()
        daily_ret = grp["close"].pct_change().dropna()
        vol = daily_ret.std() * 100          # as a percentage
        results[coin] = vol
    s = pd.Series(results)
    # Normalise: clip at 20% daily vol as "extreme"
    return (s.clip(upper=20) / 20 * 100).rename("volatility_score")


def compute_drawdown_score(market_df: pd.DataFrame) -> pd.Series:
    """
    How far is the current price from its All-Time High?
    ath_change_pct is negative → convert to positive drawdown.
    0% drawdown = 0 risk, 100% drawdown = 100 risk.
    """
    drawdown = market_df.set_index("id")["ath_change_pct"].abs()
    # Clip at 95% max drawdown for normalisation
    return (drawdown.clip(upper=95) / 95 * 100).rename("drawdown_score")


def compute_momentum_score(market_df: pd.DataFrame) -> pd.Series:
    """
    24h price change: large negative move → high risk.
    Maps [-20%, +20%] → [100, 0] risk.
    """
    change = market_df.set_index("id")["change_24h_pct"]
    # Invert: negative momentum = higher risk
    risk = (-change).clip(-20, 20)          # range [-20, +20]
    return ((risk + 20) / 40 * 100).rename("momentum_score")


def compute_liquidity_score(market_df: pd.DataFrame) -> pd.Series:
    """
    Volume-to-MarketCap ratio. Lower ratio = lower liquidity = higher risk.
    Typical healthy ratio: >5%. Below 1% is illiquid.
    """
    ratio = market_df.set_index("id")["volume_24h"] / market_df.set_index("id")["market_cap"]
    # Invert: low liquidity → high score
    risk = (1 - ratio.clip(0, 0.20) / 0.20) * 100
    return risk.rename("liquidity_score")


def compute_fear_greed_score(fg_df: pd.DataFrame) -> float:
    """
    Latest Fear & Greed value (0–100).
    Extreme Greed (100) = high complacency = higher risk.
    Extreme Fear  (0)   = panic = also risky, but differently.
    We map it as: risk peaks at both extremes (V-shape), min at 50.
    """
    latest = fg_df.sort_values("date").iloc[-1]["fg_value"]
    # V-shape: |value - 50| * 2 → 0 at 50, 100 at extremes
    return abs(latest - 50) * 2


# ─────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────

def compute_risk_scores(market_df: pd.DataFrame,
                         hist_df: pd.DataFrame,
                         fg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with individual component scores and
    a final composite RiskScore (0–100) per coin.
    """
    vol_s  = compute_volatility_score(hist_df)
    dd_s   = compute_drawdown_score(market_df)
    mom_s  = compute_momentum_score(market_df)
    liq_s  = compute_liquidity_score(market_df)
    fg_val = compute_fear_greed_score(fg_df)

    result = market_df.set_index("id")[["symbol", "name", "price_usd",
                                         "change_24h_pct", "change_7d_pct",
                                         "market_cap", "volume_24h"]].copy()

    result["volatility_score"] = vol_s
    result["drawdown_score"]   = dd_s
    result["momentum_score"]   = mom_s
    result["liquidity_score"]  = liq_s
    result["fear_greed_score"] = fg_val          # same for all coins (market-wide)

    result["risk_score"] = (
        result["volatility_score"] * WEIGHTS["volatility"] +
        result["drawdown_score"]   * WEIGHTS["drawdown"]   +
        result["momentum_score"]   * WEIGHTS["momentum"]   +
        result["liquidity_score"]  * WEIGHTS["liquidity"]  +
        result["fear_greed_score"] * WEIGHTS["fear_greed"]
    ).round(1)

    result[["risk_label", "risk_color"]] = result["risk_score"].apply(
        lambda s: pd.Series(_label(s))
    )

    return result.reset_index().sort_values("risk_score", ascending=False)


def explain_risk(row: pd.Series) -> str:
    """
    Human-readable explanation of what's driving the risk for one coin.
    """
    drivers = []
    if row["volatility_score"] > 60:
        drivers.append(f"high price volatility ({row['volatility_score']:.0f}/100)")
    if row["drawdown_score"] > 60:
        drivers.append(f"far from ATH ({row['drawdown_score']:.0f}/100 drawdown)")
    if row["momentum_score"] > 60:
        drivers.append(f"negative momentum ({row['change_24h_pct']:.1f}% in 24h)")
    if row["liquidity_score"] > 60:
        drivers.append(f"low liquidity ({row['liquidity_score']:.0f}/100)")
    if row["fear_greed_score"] > 60:
        drivers.append(f"extreme market sentiment (F&G score: {row['fear_greed_score']:.0f}/100)")

    if not drivers:
        return "✅ No major risk drivers detected."
    return "⚠️ Driven by: " + ", ".join(drivers) + "."
