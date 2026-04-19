"""
app.py  –  Crypto Risk Analyzer  |  Streamlit Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_fetcher import (
    fetch_market_overview,
    fetch_all_historical,
    fetch_fear_greed,
    TOP_COINS,
)
from risk_engine import compute_risk_scores, explain_risk, WEIGHTS
from ml_model import (
    train_models, load_models, predict_risk,
    extract_features, get_shap_values, FEATURE_COLS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Risk Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .risk-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .score-big {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stMetric > div { background: #1e1e2e; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg",
    width=60,
)
st.sidebar.title("⚙️ Settings")

selected_coins = st.sidebar.multiselect(
    "Select Coins",
    options=TOP_COINS,
    default=TOP_COINS[:6],
    format_func=lambda x: x.replace("-", " ").title(),
)

hist_days = st.sidebar.slider("Historical Window (days)", 7, 90, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Risk Weights")
w_vol  = st.sidebar.slider("Volatility",   0, 100, int(WEIGHTS["volatility"]*100))
w_dd   = st.sidebar.slider("Drawdown",     0, 100, int(WEIGHTS["drawdown"]*100))
w_mom  = st.sidebar.slider("Momentum",     0, 100, int(WEIGHTS["momentum"]*100))
w_liq  = st.sidebar.slider("Liquidity",    0, 100, int(WEIGHTS["liquidity"]*100))
w_fg   = st.sidebar.slider("Fear & Greed", 0, 100, int(WEIGHTS["fear_greed"]*100))

total_w = w_vol + w_dd + w_mom + w_liq + w_fg
if total_w == 0:
    st.sidebar.error("Weights must be non-zero!")
    st.stop()

custom_weights = {
    "volatility": w_vol / total_w,
    "drawdown":   w_dd  / total_w,
    "momentum":   w_mom / total_w,
    "liquidity":  w_liq / total_w,
    "fear_greed": w_fg  / total_w,
}

st.sidebar.caption(f"Auto-normalised. Total = {total_w}")

refresh = st.sidebar.button("🔄 Refresh Data")

# ─────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)  # 5-min cache
def load_data(coins, days):
    market  = fetch_market_overview(coins)
    hist    = fetch_all_historical(coins, days=days)
    fg      = fetch_fear_greed(limit=max(days, 30))
    return market, hist, fg

if refresh:
    st.cache_data.clear()

with st.spinner("⏳ Fetching live data from CoinGecko & Fear/Greed API…"):
    try:
        market_df, hist_df, fg_df = load_data(
            tuple(selected_coins), hist_days
        )
    except Exception as e:
        st.error(f"❌ Data fetch failed: {e}")
        st.stop()

# Override weights with sidebar values
import risk_engine
risk_engine.WEIGHTS = custom_weights

scores_df = compute_risk_scores(market_df, hist_df, fg_df)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🔐 Crypto Risk Analyzer")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  |  "
           f"Data: CoinGecko (free) + Alternative.me Fear & Greed Index")

# ─────────────────────────────────────────────
# TOP KPI ROW
# ─────────────────────────────────────────────
latest_fg  = fg_df.sort_values("date").iloc[-1]
avg_risk   = scores_df["risk_score"].mean()
most_risky = scores_df.iloc[0]
least_risky= scores_df.iloc[-1]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("😱 Fear & Greed",
            f"{latest_fg['fg_value']} – {latest_fg['fg_label']}",
            help="0=Extreme Fear, 100=Extreme Greed")
kpi2.metric("📊 Avg Portfolio Risk",
            f"{avg_risk:.1f}/100")
kpi3.metric("⬆️ Highest Risk",
            f"{most_risky['symbol']}  {most_risky['risk_score']:.0f}",
            delta=f"{most_risky['risk_label']}",
            delta_color="inverse")
kpi4.metric("⬇️ Lowest Risk",
            f"{least_risky['symbol']}  {least_risky['risk_score']:.0f}",
            delta=f"{least_risky['risk_label']}",
            delta_color="normal")

st.divider()

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏆 Risk Leaderboard", "📈 Price History", "😱 Fear & Greed", "🔍 Coin Deep Dive", "🤖 ML Prediction"]
)

# ──────────────── TAB 1: LEADERBOARD ───────────────
with tab1:
    st.subheader("Risk Leaderboard — All Selected Coins")

    # Color-coded bar chart
    fig_bar = px.bar(
        scores_df,
        x="symbol", y="risk_score",
        color="risk_score",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
        range_color=[0, 100],
        text="risk_score",
        labels={"risk_score": "Risk Score", "symbol": "Coin"},
        title="Composite Risk Score (0 = Safe, 100 = Extreme Risk)",
    )
    fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=True,
                           plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                           font_color="white", height=420)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar chart (component breakdown for top 3)
    st.subheader("📡 Component Breakdown (Top 3 Riskiest)")
    top3 = scores_df.head(3)
    categories = ["Volatility", "Drawdown", "Momentum", "Liquidity", "Fear & Greed"]
    fig_radar = go.Figure()
    for _, row in top3.iterrows():
        vals = [
            row["volatility_score"], row["drawdown_score"],
            row["momentum_score"],   row["liquidity_score"],
            row["fear_greed_score"],
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself", name=row["symbol"], opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0,100], visible=True)),
        showlegend=True, height=400,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Data table
    st.subheader("📋 Full Risk Table")
    display_cols = ["symbol", "name", "price_usd", "change_24h_pct", "change_7d_pct",
                    "risk_score", "risk_label",
                    "volatility_score", "drawdown_score", "momentum_score"]
    table_df = scores_df[display_cols].rename(columns={
        "symbol": "Symbol", "name": "Name",
        "price_usd": "Price ($)", "change_24h_pct": "24h %",
        "change_7d_pct": "7d %", "risk_score": "Risk Score",
        "risk_label": "Risk Level",
        "volatility_score": "Volatility",
        "drawdown_score": "Drawdown",
        "momentum_score": "Momentum",
    })
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                help="Composite risk score (0 = safe, 100 = extreme)",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Volatility": st.column_config.ProgressColumn(
                "Volatility", min_value=0, max_value=100, format="%d"
            ),
            "Drawdown": st.column_config.ProgressColumn(
                "Drawdown", min_value=0, max_value=100, format="%d"
            ),
            "Momentum": st.column_config.ProgressColumn(
                "Momentum", min_value=0, max_value=100, format="%d"
            ),
            "24h %": st.column_config.NumberColumn("24h %", format="%.2f%%"),
            "7d %":  st.column_config.NumberColumn("7d %",  format="%.2f%%"),
            "Price ($)": st.column_config.NumberColumn("Price ($)", format="$%.4f"),
        },
    )

# ──────────────── TAB 2: PRICE HISTORY ─────────────────
with tab2:
    st.subheader("📈 Historical Price Chart")
    if hist_df.empty:
        st.warning("No historical data loaded.")
    else:
        hist_plot = hist_df.reset_index()
        fig_hist = px.line(
            hist_plot, x="date", y="close", color="coin",
            labels={"close": "Price (USD)", "date": "Date", "coin": "Coin"},
            title=f"Closing Prices — Last {hist_days} Days",
        )
        fig_hist.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=450,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Normalised view (all start at 100)
        st.subheader("📊 Normalised Performance (base = 100)")
        norm = hist_plot.copy()
        norm["norm"] = norm.groupby("coin")["close"].transform(
            lambda x: x / x.iloc[0] * 100
        )
        fig_norm = px.line(
            norm, x="date", y="norm", color="coin",
            labels={"norm": "Normalised Price", "date": "Date"},
            title="Relative Performance (all coins start at 100)",
        )
        fig_norm.add_hline(y=100, line_dash="dash", line_color="grey")
        fig_norm.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=420,
        )
        st.plotly_chart(fig_norm, use_container_width=True)

# ──────────────── TAB 3: FEAR & GREED ──────────────────
with tab3:
    st.subheader("😱 Fear & Greed Index History")

    # Gauge for latest value
    latest_val = int(latest_fg["fg_value"])
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=latest_val,
        title={"text": f"Today: {latest_fg['fg_label']}"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#e74c3c" if latest_val < 25 else
                              "#f1c40f" if latest_val < 50 else
                              "#e67e22" if latest_val < 75 else "#2ecc71"},
            "steps": [
                {"range": [0,  25], "color": "#2c0a0a"},
                {"range": [25, 50], "color": "#2c1a0a"},
                {"range": [50, 75], "color": "#2c240a"},
                {"range": [75, 100],"color": "#0a2c0a"},
            ],
        }
    ))
    fig_gauge.update_layout(
        height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white"
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Historical line
    fig_fg = px.area(
        fg_df, x="date", y="fg_value",
        color_discrete_sequence=["#f39c12"],
        labels={"fg_value": "Fear & Greed Value", "date": "Date"},
        title=f"Fear & Greed Index — Last {len(fg_df)} Days",
    )
    fig_fg.add_hline(y=25, line_dash="dot", line_color="#e74c3c", annotation_text="Extreme Fear")
    fig_fg.add_hline(y=75, line_dash="dot", line_color="#2ecc71", annotation_text="Extreme Greed")
    fig_fg.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", height=380,
        yaxis_range=[0, 100],
    )
    st.plotly_chart(fig_fg, use_container_width=True)

# ──────────────── TAB 4: DEEP DIVE ─────────────────────
with tab4:
    st.subheader("🔍 Coin Deep Dive")
    coin_choice = st.selectbox(
        "Select a coin",
        options=scores_df["id"].tolist(),
        format_func=lambda x: scores_df.set_index("id").loc[x, "name"],
    )
    row = scores_df.set_index("id").loc[coin_choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"${row['price_usd']:,.4f}")
    c2.metric("24h Change", f"{row['change_24h_pct']:.2f}%",
              delta=f"{row['change_24h_pct']:.2f}%")
    c3.metric("7d Change",  f"{row['change_7d_pct']:.2f}%",
              delta=f"{row['change_7d_pct']:.2f}%")

    st.markdown(f"""
    <div class="risk-card">
        <p class="metric-label">Composite Risk Score</p>
        <p class="score-big" style="color:{row['risk_color']}">{row['risk_score']:.0f}</p>
        <p style="font-size:1.2rem">{row['risk_label']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(explain_risk(row))

    # Component scores breakdown
    comp_df = pd.DataFrame({
        "Component": ["Volatility", "Drawdown", "Momentum", "Liquidity", "Fear & Greed"],
        "Score": [
            row["volatility_score"], row["drawdown_score"],
            row["momentum_score"],   row["liquidity_score"],
            row["fear_greed_score"],
        ],
        "Weight": [
            custom_weights["volatility"], custom_weights["drawdown"],
            custom_weights["momentum"],   custom_weights["liquidity"],
            custom_weights["fear_greed"],
        ],
    })
    comp_df["Weighted Contribution"] = comp_df["Score"] * comp_df["Weight"]

    fig_comp = px.bar(
        comp_df, x="Component", y="Score",
        color="Score",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
        range_color=[0, 100],
        text="Score",
        title=f"Risk Component Breakdown — {row['name']}",
    )
    fig_comp.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_comp.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", height=380, showlegend=False,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Price history for this coin
    if not hist_df.empty:
        coin_hist = hist_df[hist_df["coin"] == coin_choice].reset_index()
        if not coin_hist.empty:
            fig_ch = px.area(
                coin_hist, x="date", y="close",
                title=f"{row['name']} Price — Last {hist_days} Days",
                color_discrete_sequence=["#3498db"],
            )
            fig_ch.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", height=350,
            )
            st.plotly_chart(fig_ch, use_container_width=True)

# ──────────────── TAB 5: ML PREDICTION ─────────────────
with tab5:
    st.subheader("🤖 XGBoost ML Risk Prediction")
    st.caption("12-feature model predicting both Risk Score (0–100) and Risk Level (Low/Medium/High/Extreme)")

    # ── Train / Load model ──
    @st.cache_resource(show_spinner=False)
    def get_ml_models():
        reg, clf, le = load_models()
        if reg is None:
            reg, clf, le, metrics = train_models(n_samples=8000)
            return reg, clf, le, metrics
        return reg, clf, le, {"note": "Loaded from cache"}

    with st.spinner("🧠 Training XGBoost model on 8,000 synthetic samples…"):
        reg_model, clf_model, label_enc, ml_metrics = get_ml_models()

    # Show model info
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Model",        "XGBoost")
    col_m2.metric("Features",     "12")
    col_m3.metric("Train Samples","8,000")
    if "regressor_mae" in ml_metrics:
        col_m4.metric("Score MAE", f"±{ml_metrics['regressor_mae']} pts")
    if "classifier_accuracy" in ml_metrics:
        st.info(f"✅ Classifier accuracy on held-out test set: **{ml_metrics['classifier_accuracy']}%**")

    st.divider()

    # ── Predict for all loaded coins ──
    st.subheader("📊 ML Risk Predictions — All Coins")

    ml_rows = []
    for _, mrow in market_df.iterrows():
        cid = mrow["id"]
        coin_hist_s = pd.Series(dtype=float)
        if not hist_df.empty and cid in hist_df["coin"].values:
            coin_hist_s = hist_df[hist_df["coin"] == cid]["close"]

        if coin_hist_s.empty:
            # fallback: use synthetic feature from market data only
            feats = {
                "volatility_30d":       abs(mrow["change_7d_pct"]) / 7,
                "volatility_7d":        abs(mrow["change_24h_pct"]),
                "drawdown_from_ath":    abs(mrow["ath_change_pct"]),
                "price_change_24h":     mrow["change_24h_pct"],
                "price_change_7d":      mrow["change_7d_pct"],
                "volume_to_mcap":       mrow["volume_24h"] / max(mrow["market_cap"], 1),
                "rsi_14":               50.0,
                "fear_greed":           float(fg_df.sort_values("date").iloc[-1]["fg_value"]),
                "fg_trend":             0.0,
                "mcap_rank_score":      0.5,
                "consecutive_red_days": 1 if mrow["change_24h_pct"] < 0 else 0,
                "avg_volume_ratio":     1.0,
            }
        else:
            rank = int(market_df[market_df["id"] == cid].index[0])
            feats = extract_features(
                mrow.to_dict(), coin_hist_s,
                fg_df.set_index("date")["fg_value"],
                rank, len(market_df)
            )

        pred = predict_risk(feats, reg_model, clf_model, label_enc)

        # Rule-based score for comparison
        rule_score = scores_df[scores_df["id"] == cid]["risk_score"].values
        rule_score = float(rule_score[0]) if len(rule_score) else 0.0

        ml_rows.append({
            "Symbol":         mrow["symbol"].upper(),
            "Name":           mrow["name"],
            "Rule Score":     round(rule_score, 1),
            "ML Score":       pred["ml_score"],
            "ML Label":       pred["ml_label"],
            "Confidence":     f"{pred['confidence']*100:.0f}%",
            "Delta":          round(pred["ml_score"] - rule_score, 1),
            "_color":         pred["ml_color"],
            "_feats":         feats,
            "_probs":         pred["class_probs"],
        })

    ml_df = pd.DataFrame(ml_rows)

    # Comparison bar chart: Rule vs ML
    compare_df = ml_df[["Symbol", "Rule Score", "ML Score"]].melt(
        id_vars="Symbol", var_name="Method", value_name="Score"
    )
    fig_compare = px.bar(
        compare_df, x="Symbol", y="Score", color="Method",
        barmode="group",
        color_discrete_map={"Rule Score": "#3498db", "ML Score": "#e74c3c"},
        title="Rule-Based Score vs XGBoost ML Score",
        labels={"Score": "Risk Score (0–100)"},
    )
    fig_compare.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", height=400,
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ML scores table
    display_ml = ml_df[["Symbol", "Name", "Rule Score", "ML Score", "ML Label", "Confidence", "Delta"]]
    st.dataframe(
        display_ml,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ML Score": st.column_config.ProgressColumn(
                "ML Score", min_value=0, max_value=100, format="%d"
            ),
            "Rule Score": st.column_config.ProgressColumn(
                "Rule Score", min_value=0, max_value=100, format="%d"
            ),
            "Delta": st.column_config.NumberColumn(
                "ML−Rule Delta",
                help="Positive = ML thinks it's riskier than the formula",
                format="%+.1f",
            ),
        },
    )

    st.divider()

    # ── Per-coin SHAP explainability ──
    st.subheader("🔍 Feature Impact — SHAP Explainability")
    ml_coin = st.selectbox(
        "Select coin for SHAP breakdown",
        options=ml_df["Symbol"].tolist(),
        key="ml_coin_select",
    )
    selected_ml_row = ml_df[ml_df["Symbol"] == ml_coin].iloc[0]
    feats_for_shap = selected_ml_row["_feats"]
    probs_for_coin = selected_ml_row["_probs"]

    # Confidence donut
    prob_df = pd.DataFrame({
        "Level":       list(probs_for_coin.keys()),
        "Probability": list(probs_for_coin.values()),
    })
    fig_donut = px.pie(
        prob_df, names="Level", values="Probability",
        hole=0.55,
        color="Level",
        color_discrete_map={
            "Low": "#2ecc71", "Medium": "#f1c40f",
            "High": "#e67e22", "Extreme": "#e74c3c"
        },
        title=f"{ml_coin} — Risk Level Probabilities",
    )
    fig_donut.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", height=320,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # SHAP bar chart
    shap_vals = get_shap_values(feats_for_shap, reg_model)
    if shap_vals:
        shap_df = pd.DataFrame({
            "Feature":      list(shap_vals.keys()),
            "SHAP Value":   list(shap_vals.values()),
        }).head(12)
        shap_df["Direction"] = shap_df["SHAP Value"].apply(
            lambda v: "Increases Risk" if v > 0 else "Decreases Risk"
        )
        fig_shap = px.bar(
            shap_df, x="SHAP Value", y="Feature",
            color="Direction", orientation="h",
            color_discrete_map={
                "Increases Risk": "#e74c3c",
                "Decreases Risk": "#2ecc71"
            },
            title=f"SHAP Feature Contributions — {ml_coin}",
        )
        fig_shap.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", height=420, yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("SHAP values show how much each feature pushes the ML risk score up or down from the baseline.")
    else:
        st.info("Install `shap` package to see feature contributions: `pip install shap`")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("⚡ Built with Streamlit · Data: CoinGecko API + Alternative.me · "
           "For educational/hackathon use only. Not financial advice.")
