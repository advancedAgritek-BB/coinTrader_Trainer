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
    rsi_period: int = 14,
    vol_window: int = 20,
    atr_period: int = 3,
    ema_short: int = 12,
    ema_long: int = 26,
    use_gpu: bool = False,
    log_time: bool = False,
) -> pd.DataFrame:
    """Generate technical indicator features for trading models.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing at least ``ts`` and ``price`` columns and
        ``high`` and ``low`` for ATR computation.
    rsi_period : int, optional
        Period for the RSI indicator. Defaults to ``14``.
    vol_window : int, optional
        Window size for volatility calculation. Defaults to ``20``.
    atr_period : int, optional
        Rolling period for Average True Range. Defaults to ``3``.
    ema_short : int, optional
        Span for the short EMA used in MACD. Defaults to ``12``.
    ema_long : int, optional
        Span for the long EMA used in MACD. Defaults to ``26``.
    use_gpu : bool, optional
        Whether to use ``cudf`` for GPU acceleration when available.
    log_time : bool, optional
        Print the elapsed generation time when ``True``.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with technical indicator columns added.
    """

    start_time = time.time() if log_time else None

    if 'ts' not in df.columns or 'price' not in df.columns:
        raise ValueError("DataFrame must contain 'ts' and 'price' columns")

    df = df.sort_values('ts').reset_index(drop=True).copy()

    frame: pd.DataFrame | 'cudf.DataFrame'
    if use_gpu and cudf is not None:
        frame = cudf.from_pandas(df)
    else:
        frame = df

    frame['log_ret'] = np.log(frame['price'] / frame['price'].shift(1))

    frame[f'rsi{rsi_period}'] = _rsi(frame['price'], rsi_period)

    frame[f'volatility{vol_window}'] = frame['log_ret'].rolling(
        window=vol_window, min_periods=vol_window
    ).std()

    if {'high', 'low'}.issubset(frame.columns):
        frame[f'atr{atr_period}'] = _atr(frame, atr_period)
    else:
        frame[f'atr{atr_period}'] = np.nan

    frame['ema_short'] = frame['price'].ewm(span=ema_short, adjust=False).mean()
    frame['ema_long'] = frame['price'].ewm(span=ema_long, adjust=False).mean()
    frame['macd'] = frame['ema_short'] - frame['ema_long']

    if use_gpu and cudf is not None:
        result = frame.to_pandas()
    else:
        result = frame  # type: ignore[assignment]

    result['ts'] = pd.to_datetime(result['ts'], errors='coerce')
    result = result.set_index('ts').interpolate(method='time').reset_index()

    result = result.ffill()

    nan_rows = result.isna().any(axis=1).sum()
    if len(result) > 0 and nan_rows / len(result) > 0.1:
        raise ValueError('Too many NaN values after interpolation')

    result = result.dropna()

    if log_time and start_time is not None:
        elapsed = time.time() - start_time
        print(f"feature generation took {elapsed:.3f}s")

    return result
