# 🔐 Crypto Risk Analyzer v2

> **Hackathon Project** — Theme: Blockchain  
> Assess and communicate cryptocurrency investment risks for better decision-making.

---

## ✨ Features

- **Real-time risk scoring (0–100)** per coin using 4 weighted components
- **Sentiment analysis** via VADER NLP on news headlines + Fear & Greed Index
- **Moving averages** (MA50 / MA200) with golden/death cross detection
- **Trend detection** — Bullish / Bearish / Sideways
- **Volume anomaly detection** — spike and liquidity checks
- **Risk meter UI** — animated 0–100 bar with color-coded levels
- **"Why is this risky?"** — plain-English reason list per coin
- **Buy / Hold / Avoid suggestions** based on risk level
- **Interactive price charts** with MA overlays and volatility bands
- **Fear & Greed gauge** + 30-day history
- **Adjustable risk weights** in sidebar — tune the model live
- **Fallback data** — app works even when APIs are rate-limited

---

## 🧱 Tech Stack

| Layer      | Technology                          |
|------------|-------------------------------------|
| Frontend   | Streamlit, Plotly                   |
| Backend    | Python, Pandas, NumPy               |
| Data       | CoinGecko API, Alternative.me       |
| Sentiment  | VADER Sentiment, TextBlob           |
| ML (v2)    | XGBoost, scikit-learn, SHAP         |

---

## 📁 Project Structure

```
crypto_risk_analyzer/
│
├── app.py                    ← Main Streamlit entry point
│
├── backend/
│   ├── __init__.py
│   ├── api.py                ← Clean API route functions
│   ├── data_fetcher.py       ← CoinGecko + Fear/Greed API calls
│   └── risk_engine.py        ← Risk scoring logic (0–100)
│
├── frontend/
│   ├── __init__.py
│   └── charts.py             ← All Plotly chart factories
│
├── utils/
│   ├── __init__.py
│   ├── calculations.py       ← MA, RSI, volatility, trend helpers
│   └── formatters.py         ← UI HTML/CSS formatters
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Risk Score Formula

```
Risk Score = (Volatility × 0.40) + (Sentiment × 0.20) 
           + (Market Trend × 0.20) + (Volume Anomaly × 0.20)
```

| Component       | Weight | What it measures                              |
|-----------------|--------|-----------------------------------------------|
| Volatility      | 40%    | 30-day rolling std of daily returns           |
| Sentiment       | 20%    | VADER NLP on headlines + Fear & Greed Index   |
| Market Trend    | 20%    | MA50 vs MA200 cross + 24h momentum            |
| Volume Anomaly  | 20%    | Volume spike detection + liquidity ratio      |

**Risk Levels:**

| Score   | Level   | Suggestion        |
|---------|---------|-------------------|
| 0–30    | 🟢 Low     | ✅ Buy             |
| 30–60   | 🟡 Medium  | ⏸️ Hold           |
| 60–80   | 🟠 High    | ⚠️ Caution        |
| 80–100  | 🔴 Extreme | 🚫 Avoid          |

---

## 🚀 How to Run

```bash
# 1. Clone / download the project
cd crypto_risk_analyzer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📸 Screenshots

| Dashboard | Coin Analysis | Why Risky? |
|-----------|---------------|------------|
| ![Leaderboard](screenshots/leaderboard.png) | ![Analysis](screenshots/analysis.png) | ![Explain](screenshots/explain.png) |

> _Add screenshots to a `screenshots/` folder after first run._

---

## 🗺️ Roadmap

- [ ] Whale Alert integration (large transaction monitoring)
- [ ] XGBoost ML model for predictive risk scoring
- [ ] Portfolio-level risk aggregation
- [ ] Email / Telegram alerts on risk level change
- [ ] Deployment to Streamlit Cloud
