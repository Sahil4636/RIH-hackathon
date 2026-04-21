"""
utils/calculations.py — full financial metrics
"""
import numpy as np
import pandas as pd

RISK_FREE_RATE = 0.05

def daily_returns(prices): return prices.pct_change().dropna()
def log_returns(prices): return np.log(prices/prices.shift(1)).dropna()

def holding_period_return(prices):
    if len(prices)<2: return 0.0
    return float((prices.iloc[-1]-prices.iloc[0])/prices.iloc[0]*100)

def annualised_return(prices):
    hpr=holding_period_return(prices)/100; n=len(prices)
    if n<2: return 0.0
    return float(((1+hpr)**(365/n)-1)*100)

def annualised_volatility(prices):
    ret=daily_returns(prices)
    if len(ret)<2: return 0.0
    return float(ret.std()*np.sqrt(365)*100)

def rolling_volatility(prices,window=30):
    return (daily_returns(prices).rolling(window).std()*np.sqrt(365)*100).rename("volatility")

def sharpe_ratio(prices,rfr=RISK_FREE_RATE):
    ret=daily_returns(prices)
    if len(ret)<2 or ret.std()==0: return 0.0
    excess=ret-(rfr/365)
    return float(excess.mean()/excess.std()*np.sqrt(365))

def sortino_ratio(prices,rfr=RISK_FREE_RATE):
    ret=daily_returns(prices)
    if len(ret)<2: return 0.0
    excess=ret-(rfr/365); dn=ret[ret<0]
    if len(dn)==0 or dn.std()==0: return 0.0
    return float(excess.mean()*365/(dn.std()*np.sqrt(365)))

def max_drawdown(prices):
    if len(prices)<2: return 0.0
    return float(((prices-prices.cummax())/prices.cummax()*100).min())

def drawdown_series(prices):
    return ((prices-prices.cummax())/prices.cummax()*100).rename("drawdown")

def calmar_ratio(prices):
    ar=annualised_return(prices); mdd=abs(max_drawdown(prices))
    if mdd==0: return 0.0
    return float(ar/mdd)

def beta(prices,benchmark):
    ra=daily_returns(prices); rb=daily_returns(benchmark)
    df=pd.DataFrame({"a":ra,"b":rb}).dropna()
    if len(df)<10 or df["b"].var()==0: return 1.0
    return float(df.cov().iloc[0,1]/df["b"].var())

def correlation(prices,benchmark):
    ra=daily_returns(prices); rb=daily_returns(benchmark)
    df=pd.DataFrame({"a":ra,"b":rb}).dropna()
    if len(df)<5: return 0.0
    return float(df.corr().iloc[0,1])

def value_at_risk(prices,confidence=0.95):
    ret=daily_returns(prices)*100
    if len(ret)<10: return 0.0
    return float(np.percentile(ret,(1-confidence)*100))

def conditional_var(prices,confidence=0.95):
    ret=daily_returns(prices)*100
    var=value_at_risk(prices,confidence)
    return float(ret[ret<=var].mean())

def moving_average(prices,window):
    return prices.rolling(window,min_periods=1).mean().rename(f"MA{window}")

def moving_averages(prices):
    return pd.DataFrame({"price":prices,"MA50":moving_average(prices,50),"MA200":moving_average(prices,200)})

def detect_trend(prices):
    if len(prices)<10: return "Sideways"
    ma50=float(prices.tail(50).mean())
    ma200=float(prices.tail(200).mean()) if len(prices)>=200 else ma50
    chg7=float((prices.iloc[-1]-prices.iloc[-7])/prices.iloc[-7]*100) if len(prices)>=7 else 0.0
    if ma50>ma200 and chg7>0: return "Bullish"
    if ma50<ma200 and chg7<0: return "Bearish"
    return "Sideways"

def compute_rsi(prices,period=14):
    delta=prices.diff().dropna()
    gain=delta.clip(lower=0).rolling(period).mean()
    loss=(-delta.clip(upper=0)).rolling(period).mean()
    rs=gain/loss.replace(0,np.nan)
    rsi=100-(100/(1+rs))
    return round(float(rsi.iloc[-1]),2) if not rsi.empty else 50.0

def full_metrics(prices,benchmark=None):
    btc=benchmark if benchmark is not None else prices
    return {
        "holding_period_return": round(holding_period_return(prices),2),
        "annualised_return":     round(annualised_return(prices),2),
        "annualised_volatility": round(annualised_volatility(prices),2),
        "sharpe_ratio":          round(sharpe_ratio(prices),3),
        "sortino_ratio":         round(sortino_ratio(prices),3),
        "max_drawdown":          round(max_drawdown(prices),2),
        "calmar_ratio":          round(calmar_ratio(prices),3),
        "beta":                  round(beta(prices,btc),3),
        "correlation":           round(correlation(prices,btc),3),
        "var_95":                round(value_at_risk(prices,0.95),2),
        "cvar_95":               round(conditional_var(prices,0.95),2),
        "rsi":                   round(compute_rsi(prices),1),
        "ma50":                  round(float(prices.tail(50).mean()),4),
        "ma200":                 round(float(prices.tail(200).mean()) if len(prices)>=200 else float(prices.mean()),4),
        "trend":                 detect_trend(prices),
    }

def format_large_number(n):
    if n>=1e12: return f"${n/1e12:.2f}T"
    if n>=1e9:  return f"${n/1e9:.2f}B"
    if n>=1e6:  return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"
