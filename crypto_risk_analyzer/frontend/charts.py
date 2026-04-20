"""
frontend/charts.py
------------------
All Plotly chart factory functions used by the Streamlit UI.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


DARK = dict(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")


def price_chart(hist_df: pd.DataFrame, coin_id: str,
                ma50: float = None, ma200: float = None) -> go.Figure:
    """Candlestick / area chart with MA overlays."""
    coin_hist = hist_df[hist_df["coin"] == coin_id].reset_index() \
                if "coin" in hist_df.columns else hist_df.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coin_hist["date"], y=coin_hist["close"],
        name="Price", line=dict(color="#3498db", width=2),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
    ))

    if len(coin_hist) >= 50:
        ma50_series = coin_hist["close"].rolling(50, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=coin_hist["date"], y=ma50_series,
            name="MA50", line=dict(color="#f1c40f", width=1.5, dash="dot"),
        ))
    if len(coin_hist) >= 200:
        ma200_series = coin_hist["close"].rolling(200, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=coin_hist["date"], y=ma200_series,
            name="MA200", line=dict(color="#e74c3c", width=1.5, dash="dash"),
        ))

    fig.update_layout(
        **DARK, height=380,
        title=f"{coin_id.title()} Price with Moving Averages",
        xaxis_title="Date", yaxis_title="Price (USD)",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def volatility_chart(hist_df: pd.DataFrame, coin_id: str) -> go.Figure:
    """Rolling 30-day volatility chart."""
    coin_hist = hist_df[hist_df["coin"] == coin_id].reset_index() \
                if "coin" in hist_df.columns else hist_df.reset_index()
    returns = coin_hist["close"].pct_change()
    vol = returns.rolling(30).std() * 100

    fig = go.Figure(go.Scatter(
        x=coin_hist["date"], y=vol,
        fill="tozeroy", name="30d Volatility (%)",
        line=dict(color="#e67e22", width=2),
        fillcolor="rgba(230,126,34,0.12)",
    ))
    fig.update_layout(
        **DARK, height=260,
        title="30-Day Rolling Volatility (%)",
        xaxis_title="Date", yaxis_title="Volatility %",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def risk_radar(details: dict, coin_name: str) -> go.Figure:
    categories = ["Volatility", "Sentiment", "Market Trend", "Volume Anomaly"]
    values = [
        details.get("volatility_score",     0),
        details.get("sentiment_score",      0),
        details.get("trend_score",          0),
        details.get("volume_anomaly_score", 0),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(231,76,60,0.2)",
        line=dict(color="#e74c3c", width=2),
        name=coin_name,
    ))
    fig.update_layout(
        **DARK, height=320,
        polar=dict(radialaxis=dict(range=[0, 100], visible=True,
                                   gridcolor="#333", tickfont=dict(color="#aaa"))),
        title=f"Risk Breakdown — {coin_name}",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def fear_greed_gauge(fg_value: float, fg_label: str) -> go.Figure:
    color = "#e74c3c" if fg_value < 25 else \
            "#e67e22" if fg_value < 50 else \
            "#f1c40f" if fg_value < 75 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fg_value,
        title={"text": f"Fear & Greed — {fg_label}", "font": {"color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#aaa"},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  25], "color": "#2c0a0a"},
                {"range": [25, 50], "color": "#2c1a0a"},
                {"range": [50, 75], "color": "#2c2a0a"},
                {"range": [75,100], "color": "#0a2c0a"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "value": fg_value},
        },
    ))
    fig.update_layout(**DARK, height=270, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def leaderboard_chart(risk_results: list) -> go.Figure:
    symbols = [r["symbol"] for r in risk_results]
    scores  = [r["risk_score"] for r in risk_results]
    colors  = [r["risk_color"] for r in risk_results]

    fig = go.Figure(go.Bar(
        x=symbols, y=scores,
        marker_color=colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        **DARK, height=380,
        title="Risk Score Leaderboard",
        yaxis=dict(range=[0, 110], title="Risk Score"),
        xaxis_title="Coin",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def ma_comparison_chart(hist_df: pd.DataFrame, coin_ids: list) -> go.Figure:
    """Normalised price chart for multiple coins (base = 100)."""
    fig = go.Figure()
    for cid in coin_ids:
        sub = hist_df[hist_df["coin"] == cid].reset_index() \
              if "coin" in hist_df.columns else hist_df.reset_index()
        if sub.empty:
            continue
        norm = sub["close"] / sub["close"].iloc[0] * 100
        fig.add_trace(go.Scatter(x=sub["date"], y=norm, name=cid.upper(),
                                  mode="lines"))
    fig.add_hline(y=100, line_dash="dash", line_color="#555")
    fig.update_layout(
        **DARK, height=360,
        title="Normalised Performance (base = 100)",
        xaxis_title="Date", yaxis_title="Relative Price",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
