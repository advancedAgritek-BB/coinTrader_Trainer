from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["atr", "ema", "obv", "roc", "rsi"]

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.rename(f"rsi_{period}")

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period, min_periods=period).mean()
    return atr_val.rename(f"atr_{period}")

def roc(series: pd.Series, period: int) -> pd.Series:
    roc_val = series.pct_change(period)
    return roc_val.rename(f"roc_{period}")

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    obv_val = (volume * direction).cumsum()
    return obv_val.rename("obv")
