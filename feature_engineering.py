"""Feature generation utilities used by coinTrader models."""

import pandas as pd
import numpy as np
import time

try:
    import cudf  # type: ignore
except Exception:  # pragma: no cover - optional dependency may not be installed
    cudf = None


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
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def make_features(
    df: pd.DataFrame,
    *,
    ema_short_period: int = 12,
    ema_long_period: int = 26,
    rsi_period: int = 14,
    volatility_window: int = 20,
    atr_window: int = 3,
    use_gpu: bool = False,
    log_time: bool = False,
) -> pd.DataFrame:
    """Generate technical indicator features for trading models.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing at least ``ts`` and ``price`` columns and ``high``
        and ``low`` for ATR computation.
    ema_short_period : int, optional
        Period for the short-term exponential moving average.
    ema_long_period : int, optional
        Period for the long-term exponential moving average.
    rsi_period : int, optional
        Window for computing the RSI indicator.
    volatility_window : int, optional
        Window for computing the rolling volatility of log returns.
    atr_window : int, optional
        Window for computing the Average True Range.
    use_gpu : bool, optional
        If ``True``, perform a round-trip through ``cudf`` to allow GPU acceleration.
    log_time : bool, optional
        Print the elapsed generation time when ``True``.

    Returns
    -------
    pd.DataFrame
        Data frame sorted by ``ts`` with additional feature columns:
        ``ema_short``, ``ema_long``, ``macd`` and parameterized columns
        for RSI, volatility and ATR. Missing values are forward-filled and any
        remaining ``NaN`` rows dropped.
    """

    start_time = time.time() if log_time else None

    if 'ts' not in df.columns or 'price' not in df.columns:
        raise ValueError("DataFrame must contain 'ts' and 'price' columns")

    if use_gpu:
        import cudf

        gdf = cudf.from_pandas(df)
        pl_df = gdf.to_pandas()

        df = pl_df.bfill().ffill().dropna()

        if log_time and start_time is not None:
            elapsed = time.time() - start_time
            print(f"feature generation took {elapsed:.3f}s")

        return df

    df = df.sort_values('ts').reset_index(drop=True).copy()

    # Interpolate missing values before computing indicators
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    df = df.set_index('ts').interpolate(method='time').reset_index()
    df = df.ffill()


    df['log_ret'] = np.log(df['price'] / df['price'].shift(1))

    # Exponential moving averages and MACD
    df['ema_short'] = df['price'].ewm(span=ema_short_period, adjust=False).mean()
    df['ema_long'] = df['price'].ewm(span=ema_long_period, adjust=False).mean()
    df['macd'] = df['ema_short'] - df['ema_long']

    rsi_col = f'rsi{rsi_period}'
    df[rsi_col] = _rsi(df['price'], rsi_period)

    vol_col = f'volatility{volatility_window}'
    df[vol_col] = df['log_ret'].rolling(
        window=volatility_window, min_periods=volatility_window
    ).std()

    atr_col = f'atr{atr_window}'
    if {'high', 'low'}.issubset(df.columns):
        df[atr_col] = _atr(df, atr_window)
    else:
        df[atr_col] = np.nan

    if df[[rsi_col, vol_col, atr_col]].isna().all().all():
        raise ValueError('Too many NaN values after interpolation')

    df = df.bfill().ffill()
    result = df.dropna()

    if log_time and start_time is not None:
        elapsed = time.time() - start_time
        print(f"feature generation took {elapsed:.3f}s")

    return result
