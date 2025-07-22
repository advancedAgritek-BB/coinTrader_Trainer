"""Feature generation utilities used by coinTrader models."""

import pandas as pd
import numpy as np


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return the Relative Strength Index for ``series``."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(df: pd.DataFrame, period: int = 3) -> pd.Series:
    """Return the Average True Range for ``df``."""
    high = df['high']
    low = df['low']
    close = df['price'].shift()
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features for trading models.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing at least ``ts`` and ``price`` columns and
        ``high`` and ``low`` for ATR computation.

    Returns
    -------
    pd.DataFrame
        Data frame sorted by ``ts`` with additional feature columns:
        ``log_ret``, ``rsi14``, ``volatility20`` and ``atr3``. Missing
        values are forward-filled and any remaining ``NaN`` rows dropped.
    """

    if 'ts' not in df.columns or 'price' not in df.columns:
        raise ValueError("DataFrame must contain 'ts' and 'price' columns")

    df = df.sort_values('ts').reset_index(drop=True).copy()

    df['log_ret'] = np.log(df['price'] / df['price'].shift(1))

    df['rsi14'] = _rsi(df['price'], 14)

    df['volatility20'] = df['log_ret'].rolling(window=20, min_periods=20).std()

    if {'high', 'low'}.issubset(df.columns):
        df['atr3'] = _atr(df, 3)
    else:
        df['atr3'] = np.nan

    df = df.ffill().dropna()
    return df
