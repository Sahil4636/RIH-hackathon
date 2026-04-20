"""
backend/risk_engine.py
----------------------
Risk scoring system (0–100) with:
  - Volatility     (weight 0.4)
  - Sentiment      (weight 0.2)
  - Market Trend   (weight 0.2)
  - Volume/Anomaly (weight 0.2)

Also computes: MA50, MA200, trend detection, reasons, suggestion.
"""

import numpy as np
import pandas as pd

WEIGHTS = {
    "volatility":    0.40,
    "sentiment":     0.20,
    "market_trend":  0.20,
    "volume_anomaly":0.20,
}

RISK_LEVELS = {
    (0,  30): ("Low",     "🟢", "#2ecc71"),
    (30, 60): ("Medium",  "🟡", "#f1c40f"),
    (60, 80): ("High",    "🟠", "#e67e22"),
    (80, 101):("Extreme", "🔴", "#e74c3c"),
}

SUGGESTIONS = {
    "Low":     "✅ Buy — Risk is low. Conditions look favourable.",
    "Medium":  "⏸️ Hold — Moderate risk. Monitor closely before acting.",
    "High":    "⚠️ Caution — High risk. Avoid new positions.",
    "Extreme": "🚫 Avoid — Extreme risk. Consider exiting existing positions.",
}


# ─────────────────────────────────────────────
# HELPER: risk level from score
# ─────────────────────────────────────────────
def get_risk_level(score: float) -> tuple:
    for (lo, hi), (label, icon, color) in RISK_LEVELS.items():
        if lo <= score < hi:
            return label, icon, color
    return "Extreme", "🔴", "#e74c3c"


# ─────────────────────────────────────────────
# 1. VOLATILITY SCORE (0–100)
# ─────────────────────────────────────────────
def compute_volatility_score(hist: pd.Series) -> tuple[float, float]:
    """Returns (score 0-100, raw_vol %)"""
    if len(hist) < 5:
        return 50.0, 5.0
    returns = hist.pct_change().dropna()
    vol = float(returns.std() * 100)          # daily std as %
    score = float(np.clip(vol / 15 * 100, 0, 100))   # 15% std = 100 risk
    return round(score, 2), round(vol, 4)


# ─────────────────────────────────────────────
# 2. SENTIMENT SCORE (0–100)
# ─────────────────────────────────────────────
def compute_sentiment_score(headlines: list[str],
                              fg_value: float) -> tuple[float, float]:
    """
    Returns (score 0-100, compound_sentiment -1 to 1).
    Combines VADER/TextBlob NLP on headlines + Fear & Greed Index.
    """
    nlp_score = 0.0

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        compounds = [analyzer.polarity_scores(h)["compound"] for h in headlines if h]
        nlp_score = float(np.mean(compounds)) if compounds else 0.0
    except ImportError:
        try:
            from textblob import TextBlob
            polarities = [TextBlob(h).sentiment.polarity for h in headlines if h]
            nlp_score = float(np.mean(polarities)) if polarities else 0.0
        except ImportError:
            nlp_score = 0.0

    # nlp_score: -1 (very negative) → 1 (very positive)
    # Convert: negative sentiment → high risk
    nlp_risk = (-nlp_score + 1) / 2 * 100      # 0–100

    # Fear & Greed: extreme fear (0) and extreme greed (100) both = risk
    fg_risk = abs(fg_value - 50) * 2            # 0–100

    combined = nlp_risk * 0.5 + fg_risk * 0.5
    return round(combined, 2), round(nlp_score, 4)


# ─────────────────────────────────────────────
# 3. MARKET TREND SCORE (0–100)
# ─────────────────────────────────────────────
def compute_trend_score(hist: pd.Series,
                         change_24h: float,
                         change_7d: float) -> tuple[float, str, float, float]:
    """
    Returns (score, trend_label, ma50, ma200).
    Bearish trend → high risk score.
    """
    prices = hist.sort_index()
    ma50  = float(prices.tail(50).mean())  if len(prices) >= 50  else float(prices.mean())
    ma200 = float(prices.tail(200).mean()) if len(prices) >= 200 else float(prices.mean())
    current = float(prices.iloc[-1])

    # Golden cross (MA50 > MA200) = bullish = low risk
    cross_score = 0.0
    if ma200 > 0:
        gap = (ma50 - ma200) / ma200 * 100
        cross_score = np.clip(-gap * 3, 0, 40)   # bearish gap → risk

    # Price vs MA50
    if ma50 > 0:
        below_ma50 = max(0, (ma50 - current) / ma50 * 100)
        ma_score = np.clip(below_ma50 * 4, 0, 40)
    else:
        ma_score = 20.0

    # 24h / 7d change contribution
    momentum_risk = np.clip((-change_24h + 5) * 2, 0, 20)

    score = cross_score + ma_score + momentum_risk

    if ma50 > ma200 and change_7d > 0:
        trend = "Bullish 📈"
    elif ma50 < ma200 and change_7d < 0:
        trend = "Bearish 📉"
    else:
        trend = "Sideways ↔️"

    return round(float(np.clip(score, 0, 100)), 2), trend, round(ma50, 4), round(ma200, 4)


# ─────────────────────────────────────────────
# 4. VOLUME ANOMALY SCORE (0–100)
# ─────────────────────────────────────────────
def compute_volume_score(hist_volume: pd.Series,
                          current_volume: float,
                          market_cap: float) -> tuple[float, float]:
    """
    Returns (score, volume_ratio).
    Unusual volume spikes or very low liquidity → high risk.
    """
    if len(hist_volume) >= 7:
        avg_vol = float(hist_volume.tail(30).mean())
        ratio   = current_volume / avg_vol if avg_vol > 0 else 1.0
    else:
        ratio = 1.0

    # Very high spike (>3x) or very low (<0.3x) both = risk
    if ratio > 3.0:
        spike_score = min((ratio - 3) * 20, 50)
    elif ratio < 0.3:
        spike_score = (0.3 - ratio) / 0.3 * 50
    else:
        spike_score = 0.0

    # Low liquidity (volume/mcap ratio < 1%)
    v2m = current_volume / market_cap if market_cap > 0 else 0.05
    liq_score = np.clip((0.02 - v2m) / 0.02 * 50, 0, 50)

    score = spike_score + liq_score
    return round(float(np.clip(score, 0, 100)), 2), round(ratio, 4)


# ─────────────────────────────────────────────
# COMPOSITE SCORE + REASONS + SUGGESTION
# ─────────────────────────────────────────────
def compute_full_risk(coin_id: str,
                       market_row: pd.Series,
                       hist_df: pd.DataFrame,
                       fg_df: pd.DataFrame,
                       headlines: list[str]) -> dict:
    """
    Master function — returns the full API response dict:
    {
      "coin": str,
      "risk_score": float,
      "risk_level": str,
      "reason": [str, ...],
      "suggestion": str,
      "details": { ... }
    }
    """
    # Slice history for this coin
    if not hist_df.empty and "coin" in hist_df.columns:
        coin_hist = hist_df[hist_df["coin"] == coin_id].sort_index()
    else:
        coin_hist = hist_df.sort_index() if not hist_df.empty else pd.DataFrame()

    prices  = coin_hist["close"]   if not coin_hist.empty else pd.Series([market_row.get("price_usd", 100)])
    volumes = coin_hist["volume"]  if not coin_hist.empty and "volume" in coin_hist.columns else pd.Series()

    fg_latest = float(fg_df.sort_values("date").iloc[-1]["fg_value"]) if not fg_df.empty else 50.0

    # ── Component scores ──
    vol_score,  raw_vol   = compute_volatility_score(prices)
    sent_score, sentiment = compute_sentiment_score(headlines, fg_latest)
    trend_score, trend, ma50, ma200 = compute_trend_score(
        prices,
        float(market_row.get("change_24h_pct", 0)),
        float(market_row.get("change_7d_pct",  0)),
    )
    vol_anom_score, vol_ratio = compute_volume_score(
        volumes,
        float(market_row.get("volume_24h", 0)),
        float(market_row.get("market_cap", 1)),
    )

    # ── Composite ──
    risk_score = (
        vol_score      * WEIGHTS["volatility"]     +
        sent_score     * WEIGHTS["sentiment"]      +
        trend_score    * WEIGHTS["market_trend"]   +
        vol_anom_score * WEIGHTS["volume_anomaly"]
    )
    risk_score = round(float(np.clip(risk_score, 0, 100)), 1)

    risk_label, risk_icon, risk_color = get_risk_level(risk_score)
    suggestion = SUGGESTIONS[risk_label]

    # ── Reasons ──
    reasons = _generate_reasons(
        vol_score, sent_score, trend_score, vol_anom_score,
        raw_vol, sentiment, trend, vol_ratio, fg_latest,
        float(market_row.get("change_24h_pct", 0)),
        float(market_row.get("ath_change_pct",  0)),
    )

    return {
        "coin":       coin_id,
        "symbol":     market_row.get("symbol", coin_id.upper()),
        "name":       market_row.get("name",   coin_id.title()),
        "price_usd":  market_row.get("price_usd", 0),
        "risk_score": risk_score,
        "risk_level": risk_label,
        "risk_icon":  risk_icon,
        "risk_color": risk_color,
        "reason":     reasons,
        "suggestion": suggestion,
        "details": {
            "volatility_score":    vol_score,
            "sentiment_score":     sent_score,
            "trend_score":         trend_score,
            "volume_anomaly_score":vol_anom_score,
            "raw_volatility_pct":  raw_vol,
            "sentiment_compound":  sentiment,
            "trend":               trend,
            "ma50":                ma50,
            "ma200":               ma200,
            "volume_ratio":        vol_ratio,
            "fear_greed_value":    fg_latest,
            "change_24h_pct":      market_row.get("change_24h_pct", 0),
            "change_7d_pct":       market_row.get("change_7d_pct",  0),
        },
    }


def _generate_reasons(vol_s, sent_s, trend_s, vanom_s,
                       raw_vol, sentiment, trend, vol_ratio,
                       fg_val, ch24, ath_chg) -> list[str]:
    reasons = []

    if vol_s >= 60:
        reasons.append(f"🔥 High volatility — daily price swings of {raw_vol:.1f}% (risk: {vol_s:.0f}/100)")
    elif vol_s >= 30:
        reasons.append(f"📊 Moderate volatility — {raw_vol:.1f}% daily std (risk: {vol_s:.0f}/100)")
    else:
        reasons.append(f"✅ Low volatility — stable price movement at {raw_vol:.1f}% daily std")

    if fg_val < 25:
        reasons.append(f"😱 Extreme Fear in market (F&G: {fg_val:.0f}) — panic selling possible")
    elif fg_val > 75:
        reasons.append(f"🤑 Extreme Greed in market (F&G: {fg_val:.0f}) — correction risk is high")
    else:
        reasons.append(f"😐 Neutral market sentiment (F&G: {fg_val:.0f})")

    if sentiment < -0.2:
        reasons.append(f"📰 Negative news sentiment detected (score: {sentiment:.2f})")
    elif sentiment > 0.2:
        reasons.append(f"📰 Positive news sentiment (score: {sentiment:.2f})")

    if "Bearish" in trend:
        reasons.append(f"📉 Bearish trend — MA50 below MA200, price declining")
    elif "Bullish" in trend:
        reasons.append(f"📈 Bullish trend — MA50 above MA200, upward momentum")
    else:
        reasons.append(f"↔️ Sideways trend — no clear direction")

    if ch24 < -5:
        reasons.append(f"⬇️ Sharp 24h drop of {ch24:.1f}% — short-term selling pressure")
    elif ch24 > 5:
        reasons.append(f"⬆️ Strong 24h gain of {ch24:.1f}% — momentum could reverse")

    if vol_ratio > 2.5:
        reasons.append(f"📢 Volume spike — {vol_ratio:.1f}x above 30-day average (unusual activity)")
    elif vol_ratio < 0.4:
        reasons.append(f"🔇 Low trading volume — {vol_ratio:.1f}x below average (illiquid)")

    if ath_chg < -70:
        reasons.append(f"📌 Price is {abs(ath_chg):.0f}% below All-Time High — deep drawdown")
    elif ath_chg < -30:
        reasons.append(f"📌 Price is {abs(ath_chg):.0f}% below ATH — significant correction")

    return reasons


# ─────────────────────────────────────────────
# BATCH COMPUTE
# ─────────────────────────────────────────────
def compute_all_risks(market_df: pd.DataFrame,
                       hist_df: pd.DataFrame,
                       fg_df: pd.DataFrame,
                       headlines_map: dict) -> list[dict]:
    results = []
    for _, row in market_df.iterrows():
        cid = row["id"]
        headlines = headlines_map.get(cid, [])
        result = compute_full_risk(cid, row, hist_df, fg_df, headlines)
        results.append(result)
    return sorted(results, key=lambda x: x["risk_score"], reverse=True)
