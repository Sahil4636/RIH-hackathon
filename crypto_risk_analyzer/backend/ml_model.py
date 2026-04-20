"""
ml_model.py
-----------
XGBoost-based crypto risk predictor.

Predicts TWO things:
  1. Risk Score    (0–100) — XGBoost Regressor
  2. Risk Level    (Low / Medium / High / Extreme) — XGBoost Classifier

Since we have no labelled historical dataset, we use a
"self-supervised" approach:
  - Generate synthetic training data from the rule-based
    risk engine over a wide range of market conditions
  - Train XGBoost on that data
  - The model then generalises and can weight features
    non-linearly, catching interactions the formula misses

Features (12 total):
  - volatility_30d       : 30-day rolling std of daily returns (%)
  - volatility_7d        : 7-day rolling std (short-term spike)
  - drawdown_from_ath    : % drop from all-time high (positive)
  - price_change_24h     : 24h % price change
  - price_change_7d      : 7d  % price change
  - volume_to_mcap       : volume / market cap ratio
  - rsi_14               : 14-period RSI (momentum oscillator)
  - fear_greed           : Fear & Greed Index value (0–100)
  - fg_trend             : 7-day change in Fear & Greed
  - mcap_rank_score      : normalised market cap rank (0=largest)
  - consecutive_red_days : # of consecutive days with negative returns
  - avg_volume_ratio     : current volume / 30-day avg volume
"""

import numpy as np
import pandas as pd
import os
import pickle
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
REG_PATH    = os.path.join(MODEL_DIR, "xgb_regressor.pkl")
CLF_PATH    = os.path.join(MODEL_DIR, "xgb_classifier.pkl")
ENC_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")

FEATURE_COLS = [
    "volatility_30d", "volatility_7d", "drawdown_from_ath",
    "price_change_24h", "price_change_7d", "volume_to_mcap",
    "rsi_14", "fear_greed", "fg_trend",
    "mcap_rank_score", "consecutive_red_days", "avg_volume_ratio",
]

RISK_THRESHOLDS = {"Low": 25, "Medium": 50, "High": 75, "Extreme": 100}

RISK_COLORS = {
    "Low":     "#2ecc71",
    "Medium":  "#f1c40f",
    "High":    "#e67e22",
    "Extreme": "#e74c3c",
}


# ─────────────────────────────────────────────
# HELPER: score → label
# ─────────────────────────────────────────────
def score_to_label(score: float) -> str:
    if score < 25:  return "Low"
    if score < 50:  return "Medium"
    if score < 75:  return "High"
    return "Extreme"


# ─────────────────────────────────────────────
# RSI CALCULATOR
# ─────────────────────────────────────────────
def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


# ─────────────────────────────────────────────
# FEATURE EXTRACTION from raw data
# ─────────────────────────────────────────────
def extract_features(market_row: dict,
                     hist_series: pd.Series,
                     fg_series: pd.Series,
                     mcap_rank: int,
                     total_coins: int) -> dict:
    """
    market_row  : dict with price_usd, change_24h_pct, change_7d_pct,
                  ath_change_percentage, volume_24h, market_cap
    hist_series : pd.Series of daily closing prices (index = date)
    fg_series   : pd.Series of Fear & Greed values  (index = date)
    """
    prices = hist_series.sort_index()
    returns = prices.pct_change().dropna()

    vol_30d = float(returns.tail(30).std() * 100) if len(returns) >= 5 else 5.0
    vol_7d  = float(returns.tail(7).std()  * 100) if len(returns) >= 3 else 5.0

    drawdown = abs(float(market_row.get("ath_change_pct", -50)))

    rsi = compute_rsi(prices)

    v2m = float(market_row.get("volume_24h", 0)) / max(float(market_row.get("market_cap", 1)), 1)

    fg_vals = fg_series.sort_index()
    fg_now  = float(fg_vals.iloc[-1])  if len(fg_vals) > 0 else 50.0
    fg_7ago = float(fg_vals.iloc[-7])  if len(fg_vals) >= 7 else fg_now
    fg_trend = fg_now - fg_7ago

    avg_vol_30 = returns.tail(30).abs().mean() * 100 if len(returns) >= 5 else 1.0
    avg_vol_ratio = vol_7d / max(avg_vol_30, 0.001)

    # consecutive red days
    red = 0
    for r in returns.tail(14).values[::-1]:
        if r < 0:
            red += 1
        else:
            break

    mcap_score = 1.0 - (mcap_rank / max(total_coins, 1))

    return {
        "volatility_30d":       round(vol_30d, 4),
        "volatility_7d":        round(vol_7d,  4),
        "drawdown_from_ath":    round(drawdown, 4),
        "price_change_24h":     round(float(market_row.get("change_24h_pct", 0)), 4),
        "price_change_7d":      round(float(market_row.get("change_7d_pct",  0)), 4),
        "volume_to_mcap":       round(v2m, 6),
        "rsi_14":               round(rsi, 2),
        "fear_greed":           round(fg_now, 2),
        "fg_trend":             round(fg_trend, 2),
        "mcap_rank_score":      round(mcap_score, 4),
        "consecutive_red_days": int(red),
        "avg_volume_ratio":     round(avg_vol_ratio, 4),
    }


# ─────────────────────────────────────────────
# SYNTHETIC TRAINING DATA GENERATOR
# ─────────────────────────────────────────────
def _rule_based_score(row: dict) -> float:
    """Mirrors risk_engine.py logic for label generation."""
    vol_s  = min(row["volatility_30d"] / 20 * 100, 100)
    dd_s   = min(row["drawdown_from_ath"] / 95 * 100, 100)
    mom_s  = ((-row["price_change_24h"]) + 20) / 40 * 100
    mom_s  = max(0, min(100, mom_s))
    liq_s  = (1 - min(row["volume_to_mcap"] / 0.20, 1)) * 100
    fg_s   = abs(row["fear_greed"] - 50) * 2

    # Extra non-linear penalties XGBoost will learn
    rsi_penalty = max(0, (70 - row["rsi_14"]) / 70 * 30) if row["rsi_14"] < 30 else \
                  max(0, (row["rsi_14"] - 70) / 30 * 20)
    red_penalty  = row["consecutive_red_days"] * 3
    vol_spike    = max(0, (row["volatility_7d"] - row["volatility_30d"]) * 2)

    base = (vol_s * 0.30 + dd_s * 0.25 + mom_s * 0.20 +
            liq_s * 0.15 + fg_s * 0.10)
    score = base + rsi_penalty + red_penalty * 0.5 + vol_spike * 0.3
    return float(np.clip(score, 0, 100))


def generate_training_data(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        # Realistic market distributions
        vol30  = rng.gamma(2, 2)                        # 0–20%
        vol7   = vol30 * rng.uniform(0.5, 2.5)
        dd     = rng.beta(2, 1.5) * 95                  # 0–95%
        ch24   = rng.normal(0, vol30 * 0.5)
        ch7    = rng.normal(0, vol30 * 1.5)
        v2m    = rng.beta(1.5, 8) * 0.25               # 0–25%
        rsi    = rng.uniform(10, 90)
        fg     = rng.beta(2, 2) * 100
        fg_tr  = rng.normal(0, 8)
        mcap   = rng.uniform(0, 1)
        red    = int(rng.integers(0, 10))
        avr    = rng.lognormal(0, 0.5)

        row = {
            "volatility_30d":       round(float(np.clip(vol30, 0, 25)), 4),
            "volatility_7d":        round(float(np.clip(vol7,  0, 40)), 4),
            "drawdown_from_ath":    round(float(dd), 4),
            "price_change_24h":     round(float(np.clip(ch24, -40, 40)), 4),
            "price_change_7d":      round(float(np.clip(ch7, -60, 60)), 4),
            "volume_to_mcap":       round(float(np.clip(v2m, 0, 0.25)), 6),
            "rsi_14":               round(float(rsi), 2),
            "fear_greed":           round(float(fg), 2),
            "fg_trend":             round(float(np.clip(fg_tr, -30, 30)), 2),
            "mcap_rank_score":      round(float(mcap), 4),
            "consecutive_red_days": red,
            "avg_volume_ratio":     round(float(np.clip(avr, 0.1, 5)), 4),
        }
        row["risk_score"] = _rule_based_score(row)
        row["risk_label"] = score_to_label(row["risk_score"])
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train_models(n_samples: int = 8000, save: bool = True):
    """
    Trains XGBRegressor (score) and XGBClassifier (label).
    Saves models to disk if save=True.
    Returns (regressor, classifier, label_encoder, metrics_dict).
    """
    print(f"Generating {n_samples} synthetic training samples…")
    df = generate_training_data(n_samples)

    X = df[FEATURE_COLS]
    y_reg = df["risk_score"]
    y_clf = df["risk_label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_clf)

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_enc, test_size=0.2, random_state=42
    )

    # ── Regressor ──
    print("Training XGBRegressor…")
    reg = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    reg.fit(X_tr, yr_tr,
            eval_set=[(X_te, yr_te)],
            verbose=False)

    reg_preds = reg.predict(X_te)
    mae = mean_absolute_error(yr_te, reg_preds)
    print(f"  Regressor MAE: {mae:.2f} points")

    # ── Classifier ──
    print("Training XGBClassifier…")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_tr, yc_tr,
            eval_set=[(X_te, yc_te)],
            verbose=False)

    clf_preds = clf.predict(X_te)
    acc = accuracy_score(yc_te, clf_preds)
    print(f"  Classifier Accuracy: {acc*100:.1f}%")

    metrics = {
        "regressor_mae":      round(mae, 2),
        "classifier_accuracy": round(acc * 100, 1),
        "train_samples":      len(X_tr),
        "test_samples":       len(X_te),
        "features":           FEATURE_COLS,
        "n_features":         len(FEATURE_COLS),
    }

    if save:
        with open(REG_PATH, "wb") as f: pickle.dump(reg, f)
        with open(CLF_PATH, "wb") as f: pickle.dump(clf, f)
        with open(ENC_PATH, "wb") as f: pickle.dump(le, f)
        print(f"  Models saved to {MODEL_DIR}")

    return reg, clf, le, metrics


# ─────────────────────────────────────────────
# LOAD (or train if not found)
# ─────────────────────────────────────────────
def load_models():
    if os.path.exists(REG_PATH) and os.path.exists(CLF_PATH):
        with open(REG_PATH, "rb") as f: reg = pickle.load(f)
        with open(CLF_PATH, "rb") as f: clf = pickle.load(f)
        with open(ENC_PATH, "rb") as f: le  = pickle.load(f)
        return reg, clf, le
    return None, None, None


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict_risk(features: dict,
                 reg=None, clf=None, le=None):
    """
    Given a feature dict, returns:
      {
        "ml_score":        float  (0–100),
        "ml_label":        str    (Low/Medium/High/Extreme),
        "ml_color":        str    (hex),
        "confidence":      float  (0–1, max class probability),
        "class_probs":     dict   {label: probability},
      }
    """
    if reg is None or clf is None:
        reg, clf, le = load_models()
    if reg is None:
        reg, clf, le, _ = train_models()

    X = pd.DataFrame([{c: features.get(c, 0) for c in FEATURE_COLS}])

    score = float(np.clip(reg.predict(X)[0], 0, 100))
    probs = clf.predict_proba(X)[0]
    label_idx = int(np.argmax(probs))
    label = le.inverse_transform([label_idx])[0]
    confidence = float(probs[label_idx])

    class_probs = {
        le.inverse_transform([i])[0]: round(float(p), 3)
        for i, p in enumerate(probs)
    }

    return {
        "ml_score":    round(score, 1),
        "ml_label":    label,
        "ml_color":    RISK_COLORS.get(label, "#888"),
        "confidence":  round(confidence, 3),
        "class_probs": class_probs,
    }


# ─────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
def get_shap_values(features: dict, reg=None):
    """Returns top feature contributions for one prediction."""
    try:
        import shap
        if reg is None:
            reg, _, _ = load_models()
        if reg is None:
            return {}

        X = pd.DataFrame([{c: features.get(c, 0) for c in FEATURE_COLS}])
        explainer = shap.TreeExplainer(reg)
        shap_vals = explainer.shap_values(X)[0]

        contributions = {
            feat: round(float(val), 3)
            for feat, val in zip(FEATURE_COLS, shap_vals)
        }
        return dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    except Exception:
        return {}


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    reg, clf, le, metrics = train_models(n_samples=5000)
    print("\n=== Model Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    sample = {
        "volatility_30d": 8.5, "volatility_7d": 12.0,
        "drawdown_from_ath": 65.0, "price_change_24h": -4.2,
        "price_change_7d": -9.5, "volume_to_mcap": 0.04,
        "rsi_14": 32.0, "fear_greed": 22.0, "fg_trend": -8.0,
        "mcap_rank_score": 0.9, "consecutive_red_days": 4,
        "avg_volume_ratio": 1.8,
    }
    result = predict_risk(sample, reg, clf, le)
    print(f"\n=== Sample Prediction ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    shap_vals = get_shap_values(sample, reg)
    print(f"\n=== Top SHAP Drivers ===")
    for feat, val in list(shap_vals.items())[:5]:
        direction = "↑ risk" if val > 0 else "↓ risk"
        print(f"  {feat}: {val:+.2f} ({direction})")
