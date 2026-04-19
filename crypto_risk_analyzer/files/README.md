# 🔐 Crypto Risk Analyzer

> Hackathon project — Theme: **Blockchain**  
> Problem: Assess and communicate risks in cryptocurrency investments for better decision-making.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📁 Project Structure

```
crypto_risk_analyzer/
├── app.py            ← Streamlit dashboard (UI)
├── data_fetcher.py   ← All API calls (CoinGecko + Fear & Greed)
├── risk_engine.py    ← Risk scoring logic
├── requirements.txt
└── README.md
```

---

## 📊 Data Sources (No API keys needed)

| Source | Data | Endpoint |
|--------|------|----------|
| CoinGecko (free) | Prices, market cap, volume, OHLCV | `api.coingecko.com/api/v3` |
| Alternative.me | Fear & Greed Index | `api.alternative.me/fng/` |

---

## ⚙️ Risk Score Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Volatility | 30% | 30-day rolling std of daily returns |
| Drawdown | 25% | Distance from All-Time High |
| Momentum | 20% | 24h price change direction |
| Liquidity | 15% | Volume-to-MarketCap ratio |
| Fear & Greed | 10% | Market-wide sentiment index |

**Risk Levels:**
- 🟢 0–25: Low
- 🟡 25–50: Medium  
- 🟠 50–75: High
- 🔴 75–100: Extreme

---

## 🗺️ Next Steps (for full hackathon submission)

1. **ML model** — Train XGBoost on historical drawdown data for predictive risk
2. **On-chain data** — Add Etherscan API for wallet inflows/outflows
3. **Alerts** — Email/Telegram alert when risk jumps a level
4. **Portfolio mode** — Weight-average risk across a user's holdings
5. **SHAP explainability** — Show which features drive the score most
