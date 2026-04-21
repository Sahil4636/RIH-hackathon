# RIH Hackathon - Crypto Risk Analyzer

A Streamlit-based crypto risk analytics dashboard that combines market data, sentiment, advanced risk metrics, ML predictions, and whale transaction signals into one explainable interface.

## What this project does

Crypto Risk Analyzer helps users evaluate risk across multiple coins using:
- A composite **risk score (0-100)**
- Clear **risk levels** (Low, Medium, High, Extreme)
- **Actionable suggestions** (for example: hold, caution, avoid)
- Detailed, per-coin explanations of why risk is high or low

## Implemented Features

### 1. Sidebar Controls
- Multi-asset selection
- Timeframe selection (30 days, 90 days, 1 year, 3 years, 5 years, 10 years)
- Adjustable risk weights:
  - Volatility
  - Sentiment
  - Trend
  - Volume anomaly
- Whale Alert API key input and minimum transaction size filter
- Refresh data button
- Random basket selection button

### 2. Header KPIs
- Fear & Greed Index
- Average risk score
- Highest risk coin
- Lowest risk coin
- Number of analyzed assets

### 3. Dashboard Tabs
- **Overview**
  - Normalized performance comparison
  - Risk score leaderboard
  - Quick risk cards with badge + progress + suggestion
- **Coin Analysis**
  - Detailed single-coin view
  - Risk meter and badge
  - Component-level breakdown
  - Reason-wise explanation
- **Risk Metrics**
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Calmar Ratio
  - Beta (vs BTC)
  - Correlation (vs BTC)
  - Return Measures:
    - Holding-period return
    - Annualized return
    - Average daily arithmetic return
    - Average daily log return
  - Detailed written analysis per asset
- **Price Charts**
  - Price with MA50/MA200
  - Rolling volatility
  - Drawdown
- **Fear & Greed**
  - Gauge + historical trend chart
- **Why Risky**
  - Plain-language reasons and recommendation per coin
- **ML Prediction**
  - Rule score vs XGBoost ML score
  - Label and confidence
  - SHAP feature impact visualization
- **Whale Alerts**
  - Whale activity summary
  - Volume and tx count chart
  - Live transaction feed with signal labels

## Risk Metric Formulas

Let:
- `P_t` = price at time `t`
- `r_t = (P_t - P_{t-1}) / P_{t-1}` (daily arithmetic return)
- `l_t = ln(P_t / P_{t-1})` (daily log return)
- `N` = number of trading days in period
- `R_f` = annual risk-free rate (used in code as `0.05`)
- `r_b` = benchmark return (BTC)

### Return Measures
- **Holding Period Return (HPR)**
  - `HPR = (P_end - P_start) / P_start`
- **Annualized Return**
  - `Annualized Return = ((1 + HPR)^(365/N) - 1)`
- **Average Daily Arithmetic Return**
  - `mean(r_t)`
- **Average Daily Log Return**
  - `mean(l_t)`

### Volatility
- **Annualized Volatility**
  - `sigma_ann = std(r_t) * sqrt(365)`

### Sharpe Ratio
- **Sharpe**
  - `Sharpe = (mean(r_t - R_f/365) / std(r_t)) * sqrt(365)`

### Sortino Ratio
- Downside deviation uses only negative returns:
  - `sigma_down = std(r_t | r_t < 0)`
- **Sortino**
  - `Sortino = (mean(r_t - R_f/365) * 365) / (sigma_down * sqrt(365))`

### Drawdown and Calmar
- **Drawdown at time t**
  - `DD_t = (P_t - max(P_1...P_t)) / max(P_1...P_t)`
- **Max Drawdown**
  - `MDD = min(DD_t)`
- **Calmar Ratio**
  - `Calmar = Annualized Return / |MDD|`

### Beta and Correlation (vs BTC)
- **Beta**
  - `Beta = cov(r_asset, r_benchmark) / var(r_benchmark)`
- **Correlation**
  - `Corr = corr(r_asset, r_benchmark)`

### Tail Risk (also included in app)
- **Value at Risk (VaR 95%)**
  - `VaR_95 = percentile(r_t, 5)`
- **Conditional VaR (CVaR 95%)**
  - `CVaR_95 = mean(r_t | r_t <= VaR_95)`

## Composite Risk Score (Rule Engine)

The app combines component scores into a normalized 0-100 risk score:

`Risk Score = w_v * VolatilityScore + w_s * SentimentScore + w_t * TrendScore + w_a * VolumeAnomalyScore`

Where `w_v + w_s + w_t + w_a = 1` and are adjustable from the sidebar.

## Tech Stack
- Frontend/UI: Streamlit
- Data processing: Pandas, NumPy
- Charts: Plotly
- ML: XGBoost, scikit-learn, SHAP
- APIs: CoinGecko, Alternative.me (Fear & Greed), Whale Alert

## Repository Structure

```text
crypto_risk_analyzer/
  app.py
  backend/
    api.py
    data_fetcher.py
    risk_engine.py
    ml_model.py
    whale_alert.py
  frontend/
    charts.py
  utils/
    calculations.py
    formatters.py
  requirements.txt
```

## How to run

From:
`C:\Users\sahil\Desktop\hackathon\RIH-hackathon\crypto_risk_analyzer`

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If Streamlit is missing:
```bash
python -m pip install streamlit
```

## Notes
- The app is currently maintained as **Streamlit-only** in this repository.
