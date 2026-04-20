"""
utils/formatters.py
-------------------
UI formatting helpers for Streamlit.
"""


def format_price(price: float) -> str:
    if price >= 1000:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:.4f}"
    return f"${price:.6f}"


def format_pct(value: float, show_plus: bool = True) -> str:
    sign = "+" if value > 0 and show_plus else ""
    return f"{sign}{value:.2f}%"


def risk_badge_html(label: str, color: str, icon: str) -> str:
    return f"""
    <div style="display:inline-block; background:{color}22;
                border:2px solid {color}; border-radius:8px;
                padding:6px 16px; font-size:1.1rem; font-weight:600;
                color:{color};">
        {icon} {label}
    </div>
    """


def risk_meter_html(score: float, color: str) -> str:
    """Renders an animated horizontal risk meter bar."""
    return f"""
    <div style="margin:12px 0;">
        <div style="display:flex; justify-content:space-between;
                    font-size:12px; color:#aaa; margin-bottom:4px;">
            <span>0 — Safe</span>
            <span style="font-weight:600; color:{color};">
                Score: {score:.0f} / 100
            </span>
            <span>100 — Extreme</span>
        </div>
        <div style="background:#2a2a2a; border-radius:8px;
                    height:22px; overflow:hidden;">
            <div style="width:{score}%; height:100%;
                        background:linear-gradient(90deg, #2ecc71, #f1c40f, #e67e22, {color});
                        border-radius:8px;
                        transition:width 0.6s ease;">
            </div>
        </div>
        <div style="display:flex; justify-content:space-between;
                    font-size:11px; color:#555; margin-top:2px;">
            <span>Low</span><span>Medium</span>
            <span>High</span><span>Extreme</span>
        </div>
    </div>
    """


def component_bar_html(label: str, score: float, weight: float) -> str:
    """Single component score bar for the breakdown section."""
    if score >= 60:
        bar_color = "#e74c3c"
    elif score >= 30:
        bar_color = "#f1c40f"
    else:
        bar_color = "#2ecc71"
    contrib = round(score * weight, 1)
    return f"""
    <div style="margin:6px 0;">
        <div style="display:flex; justify-content:space-between;
                    font-size:13px; margin-bottom:3px;">
            <span style="color:#ccc;">{label}
                <span style="color:#777; font-size:11px;">
                    (weight {int(weight*100)}%)
                </span>
            </span>
            <span style="color:{bar_color}; font-weight:600;">
                {score:.0f}/100 → contributes {contrib:.1f} pts
            </span>
        </div>
        <div style="background:#2a2a2a; border-radius:4px; height:10px;">
            <div style="width:{score}%; height:100%;
                        background:{bar_color}; border-radius:4px;">
            </div>
        </div>
    </div>
    """
