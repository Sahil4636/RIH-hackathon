"""
utils/calculations.py
---------------------
Shared financial calculation utilities:
  - Rolling volatility
  - Moving averages (MA50, MA200)
  - Trend detection
  - RSI
  - Normalisation helpers
"""

import numpy as np
import pandas as pd


def rolling_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
    """Returns rolling annualised volatility (%) series."""
    returns = prices.pct_change()
    return (returns.rolling(window).std() * 100).rename("volatility")


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window, min_periods=1).mean().rename(f"MA{window}")


def moving_averages(prices: pd.Series) -> pd.DataFrame:
    """Returns DataFrame with MA50 and MA200 columns."""
    return pd.DataFrame({
        "price": prices,
        "MA50":  moving_average(prices, 50),
        "MA200": moving_average(prices, 200),
    })


def detect_trend(prices: pd.Series) -> str:
    """
    Returns 'Bullish', 'Bearish', or 'Sideways' based on
    MA50 vs MA200 cross and recent momentum.
    """
    if len(prices) < 10:
        return "Sideways"
    ma50  = float(prices.tail(50).mean())
    ma200 = float(prices.tail(200).mean()) if len(prices) >= 200 else ma50
    recent_change = float((prices.iloc[-1] - prices.iloc[-7]) / prices.iloc[-7] * 100) \
                    if len(prices) >= 7 else 0.0

    if ma50 > ma200 and recent_change > 0:
        return "Bullish 📈"
    if ma50 < ma200 and recent_change < 0:
        return "Bearish 📉"
    return "Sideways ↔️"


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Returns the most recent RSI value."""
    delta = prices.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2) if not rsi.empty else 50.0


def normalise(value: float, min_val: float, max_val: float) -> float:
    """Normalise a value to 0–100 range."""
    if max_val == min_val:
        return 50.0
    return float(np.clip((value - min_val) / (max_val - min_val) * 100, 0, 100))


def format_large_number(n: float) -> str:
    if n >= 1_000_000_000_000:
        return f"${n/1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"${n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"${n/1_000_000:.2f}M"
    return f"${n:,.0f}"
