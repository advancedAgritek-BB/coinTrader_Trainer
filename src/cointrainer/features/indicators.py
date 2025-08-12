import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"rsi_{period}")

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df.get("close", df.get("price"))
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.rename(f"atr_{period}")

def ema(series: pd.Series, span: int = 12) -> pd.Series:
    ema = series.ewm(span=span, adjust=False).mean()
    return ema.rename(f"ema_{span}")

def bollinger(series: pd.Series, window: int = 20, n_std: int = 2) -> pd.Series:
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    band = (series - ma) / (n_std * std)
    return band.rename(f"boll_{window}_{n_std}")

def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    mom = series - series.shift(period)
    return mom.rename(f"mom_{period}")

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df.get("close", df.get("price"))
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period).mean()
    plus_di = (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_series) * 100
    minus_di = (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_series) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.rename(f"adx_{period}")

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    obv = (volume * direction).cumsum()
    return obv.rename("obv")

__all__ = [
    "adx",
    "atr",
    "bollinger",
    "ema",
    "momentum",
    "obv",
    "rsi",
]
