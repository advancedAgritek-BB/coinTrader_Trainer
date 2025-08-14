from __future__ import annotations
import numpy as np
import pandas as pd

def drawdown_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return dd

def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_curve(equity).min())

def sharpe(returns: pd.Series, periods_per_year: float) -> float:
    # assumes zero rf; returns are arithmetic per period
    if len(returns) < 2:
        return 0.0
    s = returns.std(ddof=0)
    if s == 0 or np.isnan(s):
        return 0.0
    return float((returns.mean() * periods_per_year) / (s * np.sqrt(periods_per_year)))

def sortino(returns: pd.Series, periods_per_year: float) -> float:
    neg = returns.copy()
    neg[neg > 0] = 0
    dn = neg.std(ddof=0)
    if dn == 0 or np.isnan(dn):
        return 0.0
    return float((returns.mean() * periods_per_year) / (dn * np.sqrt(periods_per_year)))

def cagr(equity: pd.Series, periods_per_year: float) -> float:
    if len(equity) < 2:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0])
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)

def hit_rate(ret_per_trade: pd.Series) -> float:
    if len(ret_per_trade) == 0:
        return 0.0
    wins = (ret_per_trade > 0).sum()
    return float(wins / len(ret_per_trade))

def summarize(equity: pd.Series, net_ret: pd.Series, trades_idx: pd.Index, periods_per_year: float) -> dict:
    dd = max_drawdown(equity)
    trades = len(trades_idx)
    return {
        "cagr": cagr(equity, periods_per_year),
        "sharpe": sharpe(net_ret, periods_per_year),
        "sortino": sortino(net_ret, periods_per_year),
        "max_drawdown": dd,
        "calmar": (cagr(equity, periods_per_year) / abs(dd)) if dd != 0 else 0.0,
        "turnover": float(trades / len(equity)),
        "trades": int(trades),
        "trades_per_day": float(trades / (len(equity) / (60 * 24))),  # for 1m bars
        "final_equity": float(equity.iloc[-1]),
    }
