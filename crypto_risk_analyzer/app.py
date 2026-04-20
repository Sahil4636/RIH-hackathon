"""
app.py — Crypto Risk Analyzer v2
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from backend.api import get_risk_for_coin, get_market_overview, get_historical, get_fear_greed
from backend.data_fetcher import TOP_COINS, fetch_all_historical, fetch_coin_news_headlines
from backend.risk_engine import compute_all_risks, WEIGHTS
from backend.ml_model import train_models, load_models, predict_risk, extract_features, get_shap_values
from backend.whale_alert import fetch_whale_transactions, compute_whale_signals, whale_alert_level
from frontend.charts import (
    price_chart, volatility_chart, risk_radar,
    fear_greed_gauge, leaderboard_chart, ma_comparison_chart,
)
from utils.formatters import (
    format_price, format_pct, risk_badge_html,
    risk_meter_html, component_bar_html,
)
from utils.calculations import format_large_number

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Risk Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #111827; }
  .stTabs [data-baseweb="tab"] { font-size: 14px; }
  div[data-testid="metric-container"] {
    background: #1e1e2e; border-radius: 10px; padding: 10px 14px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("🔐 Crypto Risk Analyzer")
st.sidebar.caption("v2.0 — Modular Edition")
st.sidebar.divider()

selected_coins = st.sidebar.multiselect(
    "Coins to Analyse",
    options=TOP_COINS,
    default=TOP_COINS[:6],
    format_func=lambda x: x.replace("-", " ").title(),
)
if not selected_coins:
    selected_coins = TOP_COINS[:3]

hist_days = st.sidebar.slider("Historical Window (days)", 30, 180, 90)

st.sidebar.divider()
st.sidebar.markdown("**Risk Weight Overrides**")
w_vol  = st.sidebar.slider("Volatility",    10, 70, 40) / 100
w_sent = st.sidebar.slider("Sentiment",     5,  40, 20) / 100
w_trnd = st.sidebar.slider("Market Trend",  5,  40, 20) / 100
w_vol2 = st.sidebar.slider("Volume Anomaly",5,  40, 20) / 100
total  = w_vol + w_sent + w_trnd + w_vol2
import backend.risk_engine as _re
_re.WEIGHTS = {
    "volatility":     w_vol  / total,
    "sentiment":      w_sent / total,
    "market_trend":   w_trnd / total,
    "volume_anomaly": w_vol2 / total,
}

refresh = st.sidebar.button("🔄 Refresh Data", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("### 🐋 Whale Alert")
whale_api_key = st.sidebar.text_input(
    "Whale Alert API Key",
    value="", type="password",
    placeholder="Paste free key from whale-alert.io",
    help="Leave blank to use demo data.",
)
whale_min_usd = st.sidebar.select_slider(
    "Min Transaction Size",
    options=[100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
    value=500_000,
    format_func=lambda x: f"${x/1_000_000:.1f}M",
)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_all(coins, days):
    market_df = get_market_overview(coins)
    hist_df   = fetch_all_historical(coins, days=days)
    fg_df     = get_fear_greed(limit=max(days, 30))
    headlines_map = {cid: fetch_coin_news_headlines(cid) for cid in coins[:3]}
    risk_results  = compute_all_risks(market_df, hist_df, fg_df, headlines_map)
    return market_df, hist_df, fg_df, risk_results

if refresh:
    st.cache_data.clear()

with st.spinner("⏳ Loading market data…"):
    try:
        market_df, hist_df, fg_df, risk_results = load_all(tuple(selected_coins), hist_days)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

@st.cache_data(ttl=120, show_spinner=False)
def load_whale(api_key, min_usd):
    return fetch_whale_transactions(api_key or "YOUR_API_KEY", min_usd=min_usd)

@st.cache_resource(show_spinner=False)
def get_ml_models():
    reg, clf, le = load_models()
    if reg is None:
        reg, clf, le, _ = train_models(n_samples=8000)
    return reg, clf, le

with st.spinner("🐋 Loading whale data…"):
    whale_df = load_whale(whale_api_key, whale_min_usd)
    whale_signals = compute_whale_signals(whale_df, selected_coins)

with st.spinner("🤖 Loading ML model…"):
    reg_model, clf_model, label_enc = get_ml_models()

# ─────────────────────────────────────────────
# HEADER KPIs
# ─────────────────────────────────────────────
st.title("🔐 Crypto Risk Analyzer")
st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · "
           "CoinGecko + Alternative.me · Not financial advice")

fg_latest  = fg_df.sort_values("date").iloc[-1] if not fg_df.empty else None
avg_risk   = round(sum(r["risk_score"] for r in risk_results) / max(len(risk_results), 1), 1)
top_risk   = risk_results[0]  if risk_results else {}
least_risk = risk_results[-1] if risk_results else {}

k1, k2, k3, k4 = st.columns(4)
k1.metric("😱 Fear & Greed",
          f"{int(fg_latest['fg_value'])} — {fg_latest['fg_label']}" if fg_latest is not None else "N/A")
k2.metric("📊 Avg Risk Score", f"{avg_risk}/100")
k3.metric("🔴 Most Risky",
          f"{top_risk.get('symbol','?')}  {top_risk.get('risk_score', 0):.0f}",
          delta=top_risk.get("risk_level", ""),
          delta_color="inverse")
k4.metric("🟢 Least Risky",
          f"{least_risk.get('symbol','?')}  {least_risk.get('risk_score', 0):.0f}",
          delta=least_risk.get("risk_level", ""),
          delta_color="normal")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏆 Leaderboard",
    "🔍 Coin Analysis",
    "📈 Price Charts",
    "😱 Fear & Greed",
    "⚙️ Why Is This Risky?",
    "🤖 ML Prediction",
    "🐋 Whale Alerts",
])

# ──── TAB 1: LEADERBOARD ────
with tab1:
    st.subheader("Risk Score Leaderboard")
    st.plotly_chart(leaderboard_chart(risk_results), use_container_width=True)

    st.subheader("Normalised Price Performance")
    st.plotly_chart(ma_comparison_chart(hist_df, selected_coins), use_container_width=True)

    st.subheader("📋 Full Risk Table")
    table_rows = [{
        "Symbol":     r["symbol"],
        "Name":       r["name"],
        "Price":      format_price(r["price_usd"]),
        "Risk Score": r["risk_score"],
        "Risk Level": f"{r['risk_icon']} {r['risk_level']}",
        "Trend":      r["details"].get("trend", "—"),
        "24h %":      format_pct(r["details"].get("change_24h_pct", 0)),
        "7d %":       format_pct(r["details"].get("change_7d_pct",  0)),
        "Suggestion": r["suggestion"],
    } for r in risk_results]
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True, hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=100, format="%d"),
        },
    )

# ──── TAB 2: COIN ANALYSIS ────
with tab2:
    st.subheader("🔍 Deep Dive — Single Coin")
    coin_choice = st.selectbox(
        "Select coin",
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
            st.markdown(f"**7d:**  {format_pct(d.get('change_7d_pct',  0))}")
            st.markdown(f"**Trend:** {d.get('trend', '—')}")
            st.markdown(f"**MA50:**  ${d.get('ma50', 0):,.2f}")
            st.markdown(f"**MA200:** ${d.get('ma200', 0):,.2f}")
            st.markdown(f"**RSI:**   calculated from price history")

        with c2:
            # Risk meter
            st.markdown(f"#### Risk Score")
            st.markdown(
                risk_meter_html(result["risk_score"], result["risk_color"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                risk_badge_html(result["risk_level"], result["risk_color"], result["risk_icon"]),
                unsafe_allow_html=True,
            )
            st.markdown(f"<br><b>Suggestion:</b> {result['suggestion']}", unsafe_allow_html=True)

        with c3:
            st.plotly_chart(risk_radar(result["details"], result["name"]),
                            use_container_width=True)

        st.divider()
        st.markdown("#### 📊 Component Score Breakdown")
        weights = _re.WEIGHTS
        for label, key, weight in [
            ("Volatility",     "volatility_score",     weights["volatility"]),
            ("Sentiment",      "sentiment_score",      weights["sentiment"]),
            ("Market Trend",   "trend_score",          weights["market_trend"]),
            ("Volume Anomaly", "volume_anomaly_score", weights["volume_anomaly"]),
        ]:
            st.markdown(
                component_bar_html(label, result["details"].get(key, 0), weight),
                unsafe_allow_html=True,
            )

# ──── TAB 3: PRICE CHARTS ────
with tab3:
    st.subheader("📈 Price Chart with MA50 / MA200")
    chart_coin = st.selectbox("Coin", options=selected_coins,
                               format_func=lambda x: x.replace("-", " ").title(),
                               key="chart_coin")
    if not hist_df.empty:
        st.plotly_chart(price_chart(hist_df, chart_coin), use_container_width=True)
        st.plotly_chart(volatility_chart(hist_df, chart_coin), use_container_width=True)
    else:
        st.warning("No historical data available.")

# ──── TAB 4: FEAR & GREED ────
with tab4:
    st.subheader("😱 Fear & Greed Index")
    if not fg_df.empty:
        latest_fg = fg_df.sort_values("date").iloc[-1]
        st.plotly_chart(
            fear_greed_gauge(latest_fg["fg_value"], latest_fg["fg_label"]),
            use_container_width=True,
        )
        import plotly.express as px
        fig_fg = px.area(
            fg_df, x="date", y="fg_value",
            color_discrete_sequence=["#f39c12"],
            title="Historical Fear & Greed Index",
            labels={"fg_value": "Value", "date": "Date"},
        )
        fig_fg.add_hline(y=25, line_dash="dot", line_color="#e74c3c",
                          annotation_text="Extreme Fear")
        fig_fg.add_hline(y=75, line_dash="dot", line_color="#2ecc71",
                          annotation_text="Extreme Greed")
        fig_fg.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=350, yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_fg, use_container_width=True)

# ──── TAB 5: WHY IS THIS RISKY? ────
with tab5:
    st.subheader("⚙️ Why Is This Risky? — Explainability")
    explain_coin = st.selectbox(
        "Select coin to explain",
        options=[r["coin"] for r in risk_results],
        format_func=lambda x: next((r["name"] for r in risk_results if r["coin"] == x), x),
        key="explain_coin",
    )
    result = next((r for r in risk_results if r["coin"] == explain_coin), None)

    if result:
        st.markdown(
            risk_meter_html(result["risk_score"], result["risk_color"]),
            unsafe_allow_html=True,
        )
        st.markdown(
            risk_badge_html(result["risk_level"], result["risk_color"], result["risk_icon"]),
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### 🧠 Reasons")
        for reason in result["reason"]:
            st.markdown(f"- {reason}")

        st.markdown("---")
        st.markdown(f"### 💡 Suggestion\n\n**{result['suggestion']}**")

        st.markdown("---")
        st.markdown("### 📐 Raw Metrics")
        d = result["details"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Volatility Score",       f"{d.get('volatility_score',0):.0f}/100")
            st.metric("Sentiment Score",        f"{d.get('sentiment_score',0):.0f}/100")
            st.metric("Daily Volatility",       f"{d.get('raw_volatility_pct',0):.2f}%")
            st.metric("Sentiment Compound",     f"{d.get('sentiment_compound',0):.3f}")
        with col2:
            st.metric("Trend Score",            f"{d.get('trend_score',0):.0f}/100")
            st.metric("Volume Anomaly Score",   f"{d.get('volume_anomaly_score',0):.0f}/100")
            st.metric("Volume Ratio",           f"{d.get('volume_ratio',1):.2f}x avg")
            st.metric("Fear & Greed",           f"{d.get('fear_greed_value',50):.0f}/100")


# ──── TAB 6: ML PREDICTION ────
with tab6:
    st.subheader("🤖 XGBoost ML Risk Prediction")
    st.caption("15-feature model — predicts Risk Score (0–100) and Risk Level")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Model",         "XGBoost")
    col_m2.metric("Features",      "15")
    col_m3.metric("Train Samples", "8,000")
    st.divider()

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
                "rsi_14": 50.0, "fear_greed": float(fg_df.sort_values("date").iloc[-1]["fg_value"]) if not fg_df.empty else 50.0,
                "fg_trend": 0.0, "mcap_rank_score": 0.5,
                "consecutive_red_days": 1 if mrow.get("change_24h_pct", 0) < 0 else 0,
                "avg_volume_ratio": 1.0, "whale_tx_count": 0, "whale_volume_usd": 0.0, "whale_risk_score": 0.0,
            }
        else:
            rank = int(market_df[market_df["id"] == cid].index[0])
            feats = extract_features(mrow.to_dict(), coin_hist_s,
                fg_df.set_index("date")["fg_value"] if not fg_df.empty else pd.Series(dtype=float),
                rank, len(market_df))

        if cid in whale_signals.index:
            ws = whale_signals.loc[cid]
            feats["whale_tx_count"]   = int(ws["whale_tx_count"])
            feats["whale_volume_usd"] = round(float(ws["whale_volume_usd"]) / 1_000_000, 2)
            feats["whale_risk_score"] = float(ws["whale_risk_score"])

        pred = predict_risk(feats, reg_model, clf_model, label_enc)
        rule_score = next((r["risk_score"] for r in risk_results if r["coin"] == cid), 0.0)
        ml_rows.append({
            "Symbol": mrow.get("symbol", cid).upper(), "Name": mrow.get("name", cid),
            "Rule Score": round(rule_score, 1), "ML Score": pred["ml_score"],
            "ML Label": pred["ml_label"], "Confidence": f"{pred['confidence']*100:.0f}%",
            "Delta": round(pred["ml_score"] - rule_score, 1),
            "_feats": feats, "_probs": pred["class_probs"],
        })

    ml_df = pd.DataFrame(ml_rows)
    compare_df = ml_df[["Symbol", "Rule Score", "ML Score"]].melt(id_vars="Symbol", var_name="Method", value_name="Score")
    fig_compare = px.bar(compare_df, x="Symbol", y="Score", color="Method", barmode="group",
        color_discrete_map={"Rule Score": "#3498db", "ML Score": "#e74c3c"},
        title="Rule-Based vs XGBoost ML Score")
    fig_compare.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white", height=380)
    st.plotly_chart(fig_compare, use_container_width=True)

    st.dataframe(ml_df[["Symbol","Name","Rule Score","ML Score","ML Label","Confidence","Delta"]],
        use_container_width=True, hide_index=True,
        column_config={
            "ML Score":   st.column_config.ProgressColumn("ML Score",   min_value=0, max_value=100, format="%d"),
            "Rule Score": st.column_config.ProgressColumn("Rule Score", min_value=0, max_value=100, format="%d"),
            "Delta":      st.column_config.NumberColumn("ML−Rule", format="%+.1f"),
        })

    st.divider()
    st.subheader("🔍 SHAP Explainability")
    ml_coin = st.selectbox("Coin for SHAP breakdown", options=ml_df["Symbol"].tolist(), key="ml_shap")
    sel_ml  = ml_df[ml_df["Symbol"] == ml_coin].iloc[0]
    probs   = sel_ml["_probs"]
    prob_df = pd.DataFrame({"Level": list(probs.keys()), "Probability": list(probs.values())})
    fig_donut = px.pie(prob_df, names="Level", values="Probability", hole=0.55, color="Level",
        color_discrete_map={"Low":"#2ecc71","Medium":"#f1c40f","High":"#e67e22","Extreme":"#e74c3c"},
        title=f"{ml_coin} — Risk Level Probabilities")
    fig_donut.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white", height=300)
    st.plotly_chart(fig_donut, use_container_width=True)

    shap_vals = get_shap_values(sel_ml["_feats"], reg_model)
    if shap_vals:
        shap_df = pd.DataFrame({"Feature": list(shap_vals.keys()), "SHAP Value": list(shap_vals.values())}).head(15)
        shap_df["Direction"] = shap_df["SHAP Value"].apply(lambda v: "Increases Risk" if v > 0 else "Decreases Risk")
        fig_shap = px.bar(shap_df, x="SHAP Value", y="Feature", color="Direction", orientation="h",
            color_discrete_map={"Increases Risk":"#e74c3c","Decreases Risk":"#2ecc71"},
            title=f"SHAP Feature Impact — {ml_coin}")
        fig_shap.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            height=420, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("Install `shap` to see feature contributions: `pip install shap`")

# ──── TAB 7: WHALE ALERTS ────
with tab7:
    st.subheader("🐋 Whale Alert — Large Transaction Monitor")
    if not whale_api_key:
        st.info("ℹ️ No API key — showing **demo data**. Get a free key at [whale-alert.io](https://whale-alert.io) and paste in the sidebar.")

    st.markdown("#### Whale Activity per Coin")
    wcols = st.columns(min(len(selected_coins), 5))
    for i, cid in enumerate(selected_coins[:5]):
        with wcols[i % 5]:
            if cid in whale_signals.index:
                ws = whale_signals.loc[cid]
                level, color = whale_alert_level(ws["whale_risk_score"])
                vol_m = ws["whale_volume_usd"] / 1_000_000
                st.markdown(f"""<div style="background:#1e1e2e;border:1px solid {color};border-radius:10px;
                    padding:12px;text-align:center;">
                    <div style="font-size:11px;color:#aaa;">{cid[:6].upper()}</div>
                    <div style="font-size:1.1rem;font-weight:600;color:{color};">{level}</div>
                    <div style="font-size:12px;color:#aaa;">{int(ws["whale_tx_count"])} txs · ${vol_m:.1f}M</div>
                    </div>""", unsafe_allow_html=True)

    st.divider()
    if not whale_signals.empty:
        ws_plot = whale_signals.reset_index()
        ws_plot["vol_M"] = ws_plot["whale_volume_usd"] / 1_000_000
        fig_wv = px.bar(ws_plot, x="coin_id", y="vol_M", color="whale_risk_score",
            color_continuous_scale=["#2ecc71","#f1c40f","#e67e22","#e74c3c"],
            range_color=[0,100], text="whale_tx_count",
            title="Whale Volume ($M) & Transaction Count per Coin",
            labels={"vol_M":"Volume ($M)","coin_id":"Coin"})
        fig_wv.update_traces(texttemplate="%{text} txs", textposition="outside")
        fig_wv.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white", height=380)
        st.plotly_chart(fig_wv, use_container_width=True)

    st.markdown("#### 📡 Live Transaction Feed")
    if whale_df.empty:
        st.warning("No whale transactions found.")
    else:
        feed = whale_df.copy()
        feed["amount_usd"] = feed["amount_usd"].apply(lambda x: f"${x/1_000_000:.2f}M")
        feed["amount"]     = feed["amount"].apply(lambda x: f"{x:,.0f}")
        feed["timestamp"]  = feed["timestamp"].dt.strftime("%H:%M:%S UTC")
        feed["signal"]     = feed["risk_weight"].apply(
            lambda w: "🔴 SELL PRESSURE" if w>=4 else "🟠 LARGE MOVE" if w>=3 else "🟡 TRANSFER" if w>=2 else "🟢 ACCUMULATION")
        st.dataframe(
            feed[["timestamp","symbol","amount","amount_usd","tx_type","from_type","to_type","signal","hash"]].rename(columns={
                "timestamp":"Time","symbol":"Coin","amount":"Amount","amount_usd":"USD Value",
                "tx_type":"Type","from_type":"From","to_type":"To","signal":"Signal","hash":"Tx Hash"}),
            use_container_width=True, hide_index=True)
        st.caption("🐋 Wallet→Exchange = sell pressure · Exchange→Wallet = accumulation · Refreshes every 2 min")

st.divider()
st.caption("⚡ Streamlit · CoinGecko · Alternative.me · VADER Sentiment · XGBoost · Whale Alert · For hackathon use only.")

