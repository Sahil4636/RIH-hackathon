"""
app.py - Crypto Risk Analyzer (stable UI refresh)
Run: streamlit run app.py
"""

import os
import random
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.api import get_fear_greed, get_market_overview
from backend.data_fetcher import TOP_COINS, fetch_all_historical, fetch_coin_news_headlines
from backend.ml_model import extract_features, get_shap_values, load_models, predict_risk, train_models
from backend.risk_engine import compute_all_risks
from backend.whale_alert import compute_whale_signals, fetch_whale_transactions, whale_alert_level
from frontend.charts import (
    fear_greed_gauge,
    leaderboard_chart,
    ma_comparison_chart,
    price_chart,
    risk_radar,
    volatility_chart,
)
from utils.calculations import daily_returns, full_metrics, log_returns
from utils.formatters import component_bar_html, format_pct, format_price, risk_badge_html, risk_meter_html
import backend.risk_engine as _re


def _risk_band(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "moderate"
    return "high"


def build_written_analysis(row: dict) -> str:
    vol = float(row.get("Volatility (%)", 0))
    sharpe = float(row.get("Sharpe Ratio", 0))
    sortino = float(row.get("Sortino Ratio", 0))
    mdd = abs(float(row.get("Max Drawdown (%)", 0)))
    calmar = float(row.get("Calmar Ratio", 0))
    beta = float(row.get("Beta (vs BTC)", 0))
    corr = float(row.get("Correlation (vs BTC)", 0))
    hpr = float(row.get("Holding Return (%)", 0))
    ann = float(row.get("Annualized Return (%)", 0))
    arith = float(row.get("Avg Daily Return (arith, %)", 0))
    logr = float(row.get("Avg Daily Return (log, %)", 0))

    vol_band = _risk_band(vol, 45, 85)
    dd_band = _risk_band(mdd, 20, 50)
    beta_band = _risk_band(abs(beta), 0.8, 1.3)

    risk_text = (
        f"Risk profile: Annualized volatility is **{vol:.2f}%** ({vol_band} relative risk), and max drawdown is "
        f"**-{mdd:.2f}%** ({dd_band} downside severity). Calmar ratio is **{calmar:.3f}**, indicating "
        f"how efficiently return compensated for drawdown risk."
    )
    quality_text = (
        f"Return quality: Sharpe is **{sharpe:.3f}** and Sortino is **{sortino:.3f}**. "
        f"Higher values suggest better risk-adjusted performance, especially Sortino for downside-only risk."
    )
    dependency_text = (
        f"Benchmark dependency: Beta vs BTC is **{beta:.3f}** ({beta_band} sensitivity) and correlation is "
        f"**{corr:.3f}**, showing how strongly this asset tends to move with BTC."
    )
    return_text = (
        f"Return measures: Holding-period return is **{hpr:+.2f}%**, annualized return is **{ann:+.2f}%**, "
        f"average daily arithmetic return is **{arith:+.4f}%**, and average daily log return is **{logr:+.4f}%**."
    )

    return "\n\n".join([risk_text, quality_text, dependency_text, return_text])


st.set_page_config(
    page_title="Crypto Risk Analyzer",
    page_icon="CRA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --bg: #070d19;
  --panel: #0e1526;
  --panel-2: #111b2f;
  --text: #dbe7ff;
  --muted: #8ea1c4;
  --line: #233150;
  --accent: #2f6df6;
}

[data-testid="stAppViewContainer"] { background: radial-gradient(1200px 600px at 70% -10%, #1b2b56 0%, var(--bg) 42%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1325, #081122) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--muted); }
.block-container { max-width: 1360px; padding-top: 1.2rem; }

h1, h2, h3, h4, p, span, div, label { color: var(--text); }
small, .stCaption { color: var(--muted) !important; }

[data-testid="metric-container"] {
  background: linear-gradient(180deg, var(--panel), var(--panel-2)) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  padding: 12px 14px !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: #0d1629;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 4px;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  color: var(--muted);
}

.stTabs [aria-selected="true"] {
  background: rgba(47, 109, 246, 0.18) !important;
  color: #b7cbff !important;
  border: 1px solid rgba(94, 141, 255, 0.5) !important;
}

hr { border-color: var(--line) !important; }
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Crypto Risk Analyzer")
    st.caption("BlockSage-style stable UI")
    st.divider()

    if "selected_coins_state" not in st.session_state:
        st.session_state["selected_coins_state"] = TOP_COINS[:6]

    randomize = st.button("Try a Random Crypto Basket", use_container_width=True)
    if randomize:
        sample_size = min(6, len(TOP_COINS))
        st.session_state["selected_coins_state"] = random.sample(TOP_COINS, k=sample_size)

    selected_coins = st.multiselect(
        "Assets",
        options=TOP_COINS,
        key="selected_coins_state",
        format_func=lambda x: x.replace("-", " ").title(),
    )
    if not selected_coins:
        selected_coins = TOP_COINS[:3]

    timeframe_options = {
        "30 days": 30,
        "90 days": 90,
        "1 year": 365,
        "3 years": 1095,
        "5 years": 1825,
        "10 years": 3650,
    }
    tf_label = st.select_slider("Timeframe", options=list(timeframe_options.keys()), value="90 days")
    hist_days = timeframe_options[tf_label]
    st.caption("Longer windows may load slower due to external API limits.")

    st.divider()
    st.markdown("**Risk Weights**")
    w_vol = st.slider("Volatility", 10, 70, 40)
    w_sent = st.slider("Sentiment", 5, 40, 20)
    w_trnd = st.slider("Trend", 5, 40, 20)
    w_van = st.slider("Volume Anomaly", 5, 40, 20)
    _sum_w = max(w_vol + w_sent + w_trnd + w_van, 1)
    _re.WEIGHTS = {
        "volatility": w_vol / _sum_w,
        "sentiment": w_sent / _sum_w,
        "market_trend": w_trnd / _sum_w,
        "volume_anomaly": w_van / _sum_w,
    }

    st.divider()
    st.markdown("**Whale Alert**")
    whale_api_key = st.text_input("API key", value="", type="password", placeholder="whale-alert.io")
    whale_min_usd = st.select_slider(
        "Min transaction",
        options=[100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
        value=500_000,
        format_func=lambda x: f"${x/1_000_000:.1f}M",
    )

    refresh = st.button("Refresh", use_container_width=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_all(coins, days):
    market_df = get_market_overview(list(coins))
    hist_df = fetch_all_historical(list(coins), days=days)
    fg_df = get_fear_greed(limit=max(days, 30))
    headlines_map = {cid: fetch_coin_news_headlines(cid) for cid in list(coins)[:3]}
    risk_results = compute_all_risks(market_df, hist_df, fg_df, headlines_map)
    return market_df, hist_df, fg_df, risk_results


@st.cache_data(ttl=120, show_spinner=False)
def load_whale(api_key, min_usd):
    return fetch_whale_transactions(api_key or "YOUR_API_KEY", min_usd=min_usd)


@st.cache_resource(show_spinner=False)
def get_ml_models():
    reg, clf, le = load_models()
    if reg is None:
        reg, clf, le, _ = train_models(n_samples=6000)
    return reg, clf, le


if refresh:
    st.cache_data.clear()

with st.spinner("Loading market data..."):
    try:
        market_df, hist_df, fg_df, risk_results = load_all(tuple(selected_coins), hist_days)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

with st.spinner("Loading whale data..."):
    try:
        whale_df = load_whale(whale_api_key, whale_min_usd)
        whale_signals = compute_whale_signals(whale_df, selected_coins)
    except Exception:
        whale_df = pd.DataFrame()
        whale_signals = pd.DataFrame()

with st.spinner("Loading ML model..."):
    try:
        reg_model, clf_model, label_enc = get_ml_models()
    except Exception:
        reg_model, clf_model, label_enc = None, None, None


st.markdown("# Crypto Risk Analyzer")
st.caption(f"Updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} - Not financial advice")

with st.expander("Abstract - What this analyzer measures", expanded=False):
    st.markdown(
        """
This tool quantifies crypto risk using statistical frameworks common in traditional finance, adapted for higher-volatility markets.

**Core features and explanations**
- **Volatility (sigma):** Standard deviation of daily returns, annualized.
- **Sharpe Ratio:** Excess return over risk-free rate, normalized by volatility.
- **Sortino Ratio:** Risk-adjusted return that penalizes downside deviation only.
- **Max Drawdown:** Largest peak-to-trough decline over the selected window.
- **Calmar Ratio:** Annualized return divided by max drawdown.
- **Beta and Correlation:** Sensitivity and co-movement versus benchmark assets (typically BTC/ETH in this app).
- **Return Measures:** Holding-period return, annualized return, and arithmetic vs log return context.

These metrics help compare assets consistently across timeframes, stress behavior, and relative performance.
"""
    )

fg_latest = fg_df.sort_values("date").iloc[-1] if not fg_df.empty else None
avg_risk = round(sum(r.get("risk_score", 0) for r in risk_results) / max(len(risk_results), 1), 1)
most_risky = risk_results[0] if risk_results else {}
least_risky = risk_results[-1] if risk_results else {}

fg_text = "N/A"
if fg_latest is not None:
    try:
        fg_text = f"{int(float(fg_latest['fg_value']))} - {fg_latest['fg_label']}"
    except Exception:
        fg_text = "N/A"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Fear & Greed", fg_text)
k2.metric("Average Risk", f"{avg_risk}/100")
k3.metric("Highest Risk", f"{most_risky.get('symbol', '?')} {most_risky.get('risk_score', 0):.0f}", delta=most_risky.get("risk_level", ""), delta_color="inverse")
k4.metric("Lowest Risk", f"{least_risky.get('symbol', '?')} {least_risky.get('risk_score', 0):.0f}", delta=least_risky.get("risk_level", ""), delta_color="normal")
k5.metric("Assets", len(selected_coins))

st.divider()


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Overview",
        "Coin Analysis",
        "Risk Metrics",
        "Price Charts",
        "Fear & Greed",
        "Why Risky",
        "ML Prediction",
        "Whale Alerts"
    ]
)


with tab1:
    st.plotly_chart(leaderboard_chart(risk_results), use_container_width=True)
    if not hist_df.empty:
        st.plotly_chart(ma_comparison_chart(hist_df, selected_coins), use_container_width=True)

    table_rows = [
        {
            "Symbol": r["symbol"],
            "Name": r["name"],
            "Price": format_price(r["price_usd"]),
            "Risk Score": r["risk_score"],
            "Risk Level": r["risk_level"],
            "Trend": r["details"].get("trend", "-"),
            "24h %": format_pct(r["details"].get("change_24h_pct", 0)),
            "7d %": format_pct(r["details"].get("change_7d_pct", 0)),
            "Suggestion": r["suggestion"],
        }
        for r in risk_results
    ]
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%d"),
        },
    )
    st.caption("Feature parity note: asset selection/search/edit, random basket, multi-window analysis, and risk metric comparison are all supported.")


with tab2:
    if not risk_results:
        st.info("No risk results available.")
    else:
        coin_choice = st.selectbox(
            "Asset",
            options=[r["coin"] for r in risk_results],
            format_func=lambda x: next((r["name"] for r in risk_results if r["coin"] == x), x),
        )
        result = next((r for r in risk_results if r["coin"] == coin_choice), None)

        if result:
            c1, c2, c3 = st.columns([1, 2, 1])

            with c1:
                st.markdown(f"### {result['name']} ({result['symbol']})")
                st.markdown(f"**Price:** {format_price(result['price_usd'])}")
                d = result["details"]
                st.markdown(f"**24h:** {format_pct(d.get('change_24h_pct', 0))}")
                st.markdown(f"**7d:** {format_pct(d.get('change_7d_pct', 0))}")
                st.markdown(f"**Trend:** {d.get('trend', '-')}")
                st.markdown(f"**MA50:** ${d.get('ma50', 0):,.2f}")
                st.markdown(f"**MA200:** ${d.get('ma200', 0):,.2f}")

            with c2:
                st.markdown(risk_meter_html(result["risk_score"], result["risk_color"]), unsafe_allow_html=True)
                st.markdown(
                    risk_badge_html(result["risk_level"], result["risk_color"], result["risk_icon"]),
                    unsafe_allow_html=True,
                )
                st.caption(result["suggestion"])

            with c3:
                st.plotly_chart(risk_radar(result["details"], result["name"]), use_container_width=True)

            st.markdown("#### Component Breakdown")
            for label, key, weight in [
                ("Volatility", "volatility_score", _re.WEIGHTS["volatility"]),
                ("Sentiment", "sentiment_score", _re.WEIGHTS["sentiment"]),
                ("Market Trend", "trend_score", _re.WEIGHTS["market_trend"]),
                ("Volume Anomaly", "volume_anomaly_score", _re.WEIGHTS["volume_anomaly"]),
            ]:
                st.markdown(component_bar_html(label, result["details"].get(key, 0), weight), unsafe_allow_html=True)


with tab3:
    st.markdown("#### Traditional Risk Metrics (per crypto)")
    if hist_df.empty:
        st.info("No historical data available to compute risk metrics.")
    else:
        metric_rows = []
        btc_prices = None
        if "coin" in hist_df.columns and "bitcoin" in hist_df["coin"].values:
            btc_hist = hist_df[hist_df["coin"] == "bitcoin"].sort_index()
            if not btc_hist.empty:
                btc_prices = btc_hist["close"]

        for cid in selected_coins:
            sub = hist_df[hist_df["coin"] == cid].sort_index() if "coin" in hist_df.columns else hist_df.sort_index()
            if sub.empty:
                continue
            prices = sub["close"]
            benchmark = btc_prices if btc_prices is not None else prices
            m = full_metrics(prices, benchmark)
            dret = daily_returns(prices)
            lret = log_returns(prices)
            symbol_row = market_df[market_df["id"] == cid]
            sym = symbol_row.iloc[0]["symbol"].upper() if not symbol_row.empty else cid.upper()

            metric_rows.append(
                {
                    "Asset": sym,
                    "Volatility (%)": m["annualised_volatility"],
                    "Sharpe Ratio": m["sharpe_ratio"],
                    "Sortino Ratio": m["sortino_ratio"],
                    "Max Drawdown (%)": m["max_drawdown"],
                    "Calmar Ratio": m["calmar_ratio"],
                    "Beta (vs BTC)": m["beta"],
                    "Correlation (vs BTC)": m["correlation"],
                    "Holding Return (%)": m["holding_period_return"],
                    "Annualized Return (%)": m["annualised_return"],
                    "Avg Daily Return (arith, %)": round(float(dret.mean() * 100), 4) if len(dret) else 0.0,
                    "Avg Daily Return (log, %)": round(float(lret.mean() * 100), 4) if len(lret) else 0.0,
                }
            )

        if not metric_rows:
            st.info("Could not compute metrics for selected assets.")
        else:
            metrics_df = pd.DataFrame(metric_rows)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            st.markdown("#### Detailed Written Analysis")
            for row in metric_rows:
                with st.expander(f"{row['Asset']} - Detailed Analysis", expanded=False):
                    st.markdown(build_written_analysis(row))

            st.markdown("#### Metric Meanings")
            st.markdown(
                """
- **Sharpe Ratio:** Excess return per unit of total volatility.
- **Sortino Ratio:** Excess return per unit of downside volatility only.
- **Max Drawdown:** Largest peak-to-trough loss in the selected period.
- **Calmar Ratio:** Annualized return divided by max drawdown.
- **Beta & Correlation:** Sensitivity and co-movement versus BTC benchmark.
- **Return Measures:** Holding-period, annualized, arithmetic daily and log daily returns.
"""
            )


with tab4:
    chart_coin = st.selectbox(
        "Asset",
        options=selected_coins,
        format_func=lambda x: x.replace("-", " ").title(),
        key="chart_coin",
    )
    if hist_df.empty:
        st.warning("No historical data available.")
    else:
        st.plotly_chart(price_chart(hist_df, chart_coin), use_container_width=True)
        st.plotly_chart(volatility_chart(hist_df, chart_coin), use_container_width=True)

        coin_hist = hist_df[hist_df["coin"] == chart_coin].sort_index() if "coin" in hist_df.columns else hist_df.sort_index()
        if not coin_hist.empty:
            prices = coin_hist["close"]
            metrics = full_metrics(prices)
            dret = daily_returns(prices)
            lret = log_returns(prices)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Holding Return", f"{metrics['holding_period_return']:+.2f}%")
            c2.metric("Annualized Return", f"{metrics['annualised_return']:+.2f}%")
            c3.metric("Avg Daily Return (arith)", f"{(dret.mean()*100 if len(dret) else 0):+.3f}%")
            c4.metric("Avg Daily Return (log)", f"{(lret.mean()*100 if len(lret) else 0):+.3f}%")

            with st.expander("Feature explanations (from BlockSage abstract)", expanded=False):
                st.markdown(
                    """
- **Volatility:** Annualized standard deviation of daily returns.
- **Sharpe:** Excess return per unit of total volatility.
- **Sortino:** Excess return per unit of downside volatility only.
- **Max Drawdown:** Worst peak-to-trough decline during the period.
- **Calmar:** Annualized return divided by max drawdown.
- **Beta/Correlation:** Market sensitivity and relationship to benchmark behavior.
- **Return measures:** Holding-period, annualized, and arithmetic/log return lens.
"""
                )


with tab5:
    if fg_df.empty:
        st.info("Fear & Greed data unavailable.")
    else:
        latest_fg = fg_df.sort_values("date").iloc[-1]
        st.plotly_chart(fear_greed_gauge(latest_fg["fg_value"], latest_fg["fg_label"]), use_container_width=True)

        fig_fg = px.area(
            fg_df,
            x="date",
            y="fg_value",
            color_discrete_sequence=["#2f6df6"],
            title="Historical Fear & Greed Index",
            labels={"fg_value": "Value", "date": "Date"},
        )
        fig_fg.add_hline(y=25, line_dash="dot", line_color="#ef4444", annotation_text="Extreme Fear")
        fig_fg.add_hline(y=75, line_dash="dot", line_color="#22c55e", annotation_text="Extreme Greed")
        fig_fg.update_layout(plot_bgcolor="#0e1526", paper_bgcolor="#0e1526", font_color="#dbe7ff", height=340, yaxis_range=[0, 100])
        st.plotly_chart(fig_fg, use_container_width=True)


with tab6:
    if not risk_results:
        st.info("No explainability data available.")
    else:
        explain_coin = st.selectbox(
            "Asset",
            options=[r["coin"] for r in risk_results],
            format_func=lambda x: next((r["name"] for r in risk_results if r["coin"] == x), x),
            key="explain_coin",
        )
        result = next((r for r in risk_results if r["coin"] == explain_coin), None)

        if result:
            st.markdown(risk_meter_html(result["risk_score"], result["risk_color"]), unsafe_allow_html=True)
            st.markdown(
                risk_badge_html(result["risk_level"], result["risk_color"], result["risk_icon"]),
                unsafe_allow_html=True,
            )
            st.markdown("#### Reasons")
            for reason in result["reason"]:
                st.markdown(f"- {reason}")
            st.markdown(f"#### Suggestion\n{result['suggestion']}")


with tab7:
    st.caption("XGBoost risk model")
    if reg_model is None or clf_model is None or label_enc is None:
        st.info("ML model unavailable in current environment.")
    else:
        ml_rows = []
        for _, mrow in market_df.iterrows():
            cid = mrow["id"]
            coin_hist_s = pd.Series(dtype=float)
            if not hist_df.empty and "coin" in hist_df.columns and cid in hist_df["coin"].values:
                coin_hist_s = hist_df[hist_df["coin"] == cid]["close"]

            if coin_hist_s.empty:
                feats = {
                    "volatility_30d": abs(mrow.get("change_7d_pct", 0)) / 7,
                    "volatility_7d": abs(mrow.get("change_24h_pct", 0)),
                    "drawdown_from_ath": abs(mrow.get("ath_change_pct", 0)),
                    "price_change_24h": mrow.get("change_24h_pct", 0),
                    "price_change_7d": mrow.get("change_7d_pct", 0),
                    "volume_to_mcap": mrow.get("volume_24h", 0) / max(mrow.get("market_cap", 1), 1),
                    "rsi_14": 50.0,
                    "fear_greed": float(fg_df.sort_values("date").iloc[-1]["fg_value"]) if not fg_df.empty else 50.0,
                    "fg_trend": 0.0,
                    "mcap_rank_score": 0.5,
                    "consecutive_red_days": 1 if mrow.get("change_24h_pct", 0) < 0 else 0,
                    "avg_volume_ratio": 1.0,
                    "whale_tx_count": 0,
                    "whale_volume_usd": 0.0,
                    "whale_risk_score": 0.0,
                }
            else:
                rank = int(market_df[market_df["id"] == cid].index[0])
                feats = extract_features(
                    mrow.to_dict(),
                    coin_hist_s,
                    fg_df.set_index("date")["fg_value"] if not fg_df.empty else pd.Series(dtype=float),
                    rank,
                    len(market_df),
                )

            if not whale_signals.empty and cid in whale_signals.index:
                ws = whale_signals.loc[cid]
                feats["whale_tx_count"] = int(ws["whale_tx_count"])
                feats["whale_volume_usd"] = round(float(ws["whale_volume_usd"]) / 1_000_000, 2)
                feats["whale_risk_score"] = float(ws["whale_risk_score"])

            pred = predict_risk(feats, reg_model, clf_model, label_enc)
            rule_score = next((r["risk_score"] for r in risk_results if r["coin"] == cid), 0.0)
            ml_rows.append(
                {
                    "Symbol": mrow.get("symbol", cid).upper(),
                    "Name": mrow.get("name", cid),
                    "Rule Score": round(rule_score, 1),
                    "ML Score": pred["ml_score"],
                    "ML Label": pred["ml_label"],
                    "Confidence": f"{pred['confidence']*100:.0f}%",
                    "Delta": round(pred["ml_score"] - rule_score, 1),
                    "_feats": feats,
                    "_probs": pred["class_probs"],
                }
            )

        ml_df = pd.DataFrame(ml_rows)
        compare_df = ml_df[["Symbol", "Rule Score", "ML Score"]].melt(id_vars="Symbol", var_name="Method", value_name="Score")
        fig_compare = px.bar(compare_df, x="Symbol", y="Score", color="Method", barmode="group")
        fig_compare.update_layout(plot_bgcolor="#0e1526", paper_bgcolor="#0e1526", font_color="#dbe7ff", height=340)
        st.plotly_chart(fig_compare, use_container_width=True)

        st.dataframe(
            ml_df[["Symbol", "Name", "Rule Score", "ML Score", "ML Label", "Confidence", "Delta"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### SHAP Explainability")
        ml_coin = st.selectbox("Coin", options=ml_df["Symbol"].tolist(), key="ml_shap")
        sel_ml = ml_df[ml_df["Symbol"] == ml_coin].iloc[0]
        probs = sel_ml["_probs"]
        prob_df = pd.DataFrame({"Level": list(probs.keys()), "Probability": list(probs.values())})
        fig_donut = px.pie(prob_df, names="Level", values="Probability", hole=0.55, title=f"{ml_coin} class probabilities")
        fig_donut.update_layout(plot_bgcolor="#0e1526", paper_bgcolor="#0e1526", font_color="#dbe7ff", height=300)
        st.plotly_chart(fig_donut, use_container_width=True)

        shap_vals = get_shap_values(sel_ml["_feats"], reg_model)
        if shap_vals:
            shap_df = pd.DataFrame({"Feature": list(shap_vals.keys()), "SHAP Value": list(shap_vals.values())}).head(15)
            shap_df["Direction"] = shap_df["SHAP Value"].apply(lambda v: "Increases Risk" if v > 0 else "Decreases Risk")
            fig_shap = px.bar(shap_df, x="SHAP Value", y="Feature", color="Direction", orientation="h")
            fig_shap.update_layout(plot_bgcolor="#0e1526", paper_bgcolor="#0e1526", font_color="#dbe7ff", height=420)
            st.plotly_chart(fig_shap, use_container_width=True)


with tab8:
    st.markdown("#### Whale Alert Monitor")
    if not whale_api_key:
        st.info("No API key set. Showing demo mode.")

    if not whale_signals.empty:
        ws_plot = whale_signals.reset_index()
        ws_plot["vol_M"] = ws_plot["whale_volume_usd"] / 1_000_000
        fig_wv = px.bar(
            ws_plot,
            x="coin_id",
            y="vol_M",
            color="whale_risk_score",
            text="whale_tx_count",
            color_continuous_scale=["#22c55e", "#f59e0b", "#f97316", "#ef4444"],
            range_color=[0, 100],
            title="Whale Volume ($M) and Transaction Count",
            labels={"vol_M": "Volume ($M)", "coin_id": "Coin"},
        )
        fig_wv.update_traces(texttemplate="%{text} tx", textposition="outside")
        fig_wv.update_layout(plot_bgcolor="#0e1526", paper_bgcolor="#0e1526", font_color="#dbe7ff", height=340)
        st.plotly_chart(fig_wv, use_container_width=True)

    if whale_df.empty:
        st.warning("No whale transactions found.")
    else:
        feed = whale_df.copy()
        feed["amount_usd"] = feed["amount_usd"].apply(lambda x: f"${x/1_000_000:.2f}M")
        feed["amount"] = feed["amount"].apply(lambda x: f"{x:,.0f}")
        feed["timestamp"] = pd.to_datetime(feed["timestamp"], errors="coerce", utc=True).dt.strftime("%H:%M:%S UTC")
        feed["signal"] = feed["risk_weight"].apply(
            lambda w: "Sell Pressure" if w >= 4 else "Large Move" if w >= 3 else "Transfer" if w >= 2 else "Accumulation"
        )
        st.dataframe(
            feed[["timestamp", "symbol", "amount", "amount_usd", "tx_type", "from_type", "to_type", "signal", "hash"]].rename(
                columns={
                    "timestamp": "Time",
                    "symbol": "Coin",
                    "amount": "Amount",
                    "amount_usd": "USD Value",
                    "tx_type": "Type",
                    "from_type": "From",
                    "to_type": "To",
                    "signal": "Signal",
                    "hash": "Tx Hash",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


st.divider()
st.caption("Streamlit · CoinGecko · Alternative.me · XGBoost · Whale Alert")
