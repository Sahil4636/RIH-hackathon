"""
app.py — Crypto Risk Analyzer v2
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from datetime import datetime

from backend.api import get_risk_for_coin, get_market_overview, get_historical, get_fear_greed
from backend.data_fetcher import TOP_COINS, fetch_all_historical, fetch_coin_news_headlines
from backend.risk_engine import compute_all_risks, WEIGHTS
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Leaderboard",
    "🔍 Coin Analysis",
    "📈 Price Charts",
    "😱 Fear & Greed",
    "⚙️ Why Is This Risky?",
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

st.divider()
st.caption("⚡ Built with Streamlit · CoinGecko · Alternative.me · VADER Sentiment · "
           "For hackathon/educational use only. Not financial advice.")
