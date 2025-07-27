"""Feature generation utilities used by coinTrader models."""

import os
import platform
import time
from utils import timed
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from cache_utils import load_cached_features, store_cached_features

try:  # optional dependency for GPU acceleration
    import jax.numpy as jnp  # type: ignore
except Exception:  # pragma: no cover - jax may be absent
    jnp = None  # type: ignore
import numba

logger = logging.getLogger(__name__)


def has_rocm() -> bool:
    """Return ``True`` if ROCm is available on Windows."""
    if platform.system() != "Windows":
        return False
    return "ROCM_PATH" in os.environ or os.path.exists(
        "C:\\Program Files\\AMD\\ROCm"
    )


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
    high = df["high"]
    low = df["low"]
    close = df["price"].shift()
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


@numba.njit
def _rsi_nb(arr: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(arr)
    rsi = np.empty(n)
    rsi[:] = np.nan
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        diff = arr[i] - arr[i - 1]
        if diff > 0:
            gains[i] = diff
            losses[i] = 0.0
        else:
            gains[i] = 0.0
            losses[i] = -diff
    avg_gain = np.sum(gains[1 : period + 1]) / period
    avg_loss = np.sum(losses[1 : period + 1]) / period
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


@numba.njit
def _atr_nb(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(high)
    atr = np.empty(n)
    atr[:] = np.nan
    tr = np.zeros(n)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    atr_val = np.sum(tr[1 : period + 1]) / period
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        atr[i] = atr_val
    return atr


@numba.njit
def _adx_nb(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(high)
    adx = np.empty(n)
    adx[:] = np.nan
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    atr_val = np.sum(tr[1 : period + 1]) / period
    pdm = np.sum(plus_dm[1 : period + 1]) / period
    mdm = np.sum(minus_dm[1 : period + 1]) / period

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        pdm = (pdm * (period - 1) + plus_dm[i]) / period
        mdm = (mdm * (period - 1) + minus_dm[i]) / period

        if atr_val != 0.0:
            plus_di[i] = 100.0 * pdm / atr_val
            minus_di[i] = 100.0 * mdm / atr_val
        den = plus_di[i] + minus_di[i]
        if den != 0.0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / den
        else:
            dx[i] = 0.0

    adx_val = np.sum(dx[period + 1 : 2 * period + 1]) / period
    for i in range(2 * period + 1, n):
        adx_val = (adx_val * (period - 1) + dx[i]) / period
        adx[i] = adx_val

    return adx


@numba.njit
def _obv_nb(price: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = len(price)
    obv = np.zeros(n)
    for i in range(1, n):
        diff = price[i] - price[i - 1]
        direction = 0.0
        if diff > 0:
            direction = 1.0
        elif diff < 0:
            direction = -1.0
        obv[i] = obv[i - 1] + direction * volume[i]
    return obv


def _bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return Bollinger Band series for ``series``."""
    mid = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def _momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Return the momentum indicator for ``series``."""
    return series - series.shift(period)


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Average Directional Index for ``df``."""
    high = df["high"]
    low = df["low"]
    close = df["price"].shift()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


def _obv(df: pd.DataFrame) -> pd.Series:
    """Return the On-Balance Volume for ``df``."""
    price_diff = df["price"].diff().fillna(0)
    direction = np.sign(price_diff)
    volume = df.get("volume", pd.Series(0, index=df.index)).fillna(0)
    obv = (direction * volume).cumsum()
    return obv


def _compute_features_pandas(
    df: pd.DataFrame,
    ema_short_period: int,
    ema_long_period: int,
    rsi_period: int,
    volatility_window: int,
    atr_window: int,
    bollinger_window: int,
    bollinger_std: float,
    momentum_period: int,
    adx_period: int,
    use_numba: bool = False,
) -> tuple[pd.DataFrame, str, str, str]:
    df = df.sort_values("ts").reset_index(drop=True).copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.set_index("ts").interpolate(method="time").reset_index()
    df = df.ffill()

    df["log_ret"] = np.log(df["price"] / df["price"].shift(1))
    df["ema_short"] = df["price"].ewm(span=ema_short_period, adjust=False).mean()
    df["ema_long"] = df["price"].ewm(span=ema_long_period, adjust=False).mean()
    df["macd"] = df["ema_short"] - df["ema_long"]

    mid, upper, lower = _bollinger_bands(df["price"], bollinger_window, bollinger_std)
    df["bol_mid"] = mid
    df["bol_upper"] = upper
    df["bol_lower"] = lower

    mom_col = f"momentum_{momentum_period}"
    df[mom_col] = _momentum(df["price"], momentum_period)

    rsi_col = f"rsi{rsi_period}"
    if use_numba:
        df[rsi_col] = pd.Series(
            _rsi_nb(df["price"].to_numpy(), rsi_period), index=df.index
        )
    else:
        df[rsi_col] = _rsi(df["price"], rsi_period)

    vol_col = f"volatility{volatility_window}"
    df[vol_col] = df["log_ret"].rolling(
        window=volatility_window, min_periods=volatility_window
    ).std()

    atr_col = f"atr{atr_window}"
    if {"high", "low"}.issubset(df.columns):
        if use_numba:
            df[atr_col] = pd.Series(
                _atr_nb(
                    df["high"].to_numpy(),
                    df["low"].to_numpy(),
                    df["price"].to_numpy(),
                    atr_window,
                ),
                index=df.index,
            )
        else:
            df[atr_col] = _atr(df, atr_window)
    else:
        df[atr_col] = np.nan

    adx_col = f"adx_{adx_period}"
    if {"high", "low"}.issubset(df.columns):
        if use_numba:
            df[adx_col] = pd.Series(
                _adx_nb(
                    df["high"].to_numpy(),
                    df["low"].to_numpy(),
                    df["price"].to_numpy(),
                    adx_period,
                ),
                index=df.index,
            )
        else:
            df[adx_col] = _adx(df, adx_period)
    else:
        df[adx_col] = np.nan

    if "volume" in df.columns:
        if use_numba:
            df["obv"] = pd.Series(
                _obv_nb(df["price"].to_numpy(), df["volume"].fillna(0).to_numpy()),
                index=df.index,
            )
        else:
            df["obv"] = _obv(df)
    else:
        df["obv"] = np.nan

    return df, rsi_col, vol_col, atr_col


@timed
def make_features(
    df: pd.DataFrame,
    *,
    ema_short_period: int = 12,
    ema_long_period: int = 26,
    rsi_period: int = 14,
    volatility_window: int = 20,
    atr_window: int = 3,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    momentum_period: int = 10,
    adx_period: int = 14,
    use_gpu: bool = False,
    return_threshold: float = 0.01,
    use_modin: bool = False,
    redis_client: Optional[Any] = None,
    cache_key: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    use_dask: bool = False,

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
    bollinger_window : int, optional
        Lookback window for Bollinger Band calculations.
    bollinger_std : float, optional
        Standard deviation multiplier for Bollinger Bands.
    momentum_period : int, optional
        Period for the momentum indicator.
    adx_period : int, optional
        Period for the Average Directional Index.
    use_gpu : bool, optional
        If ``True``, perform a round-trip through ``cudf`` to allow GPU acceleration.
        When enabled, JAX and Numba are used for faster indicator computation.
        The underlying ``_compute_features_pandas`` call receives ``use_numba=True``
        to activate the Numba-optimised implementations.
        If ``True``, calculations are accelerated on the GPU using JAX and Numba.
        OpenCL/ROCm are used when available to provide GPU support.
        When ``True`` Numba accelerated functions operate on NumPy arrays
        for faster computation.
    log_time : bool, optional
        Print the elapsed generation time when ``True``.
    return_threshold : float, optional
        Threshold used to generate the ``target`` column when it is missing.
    use_modin : bool, optional
        Convert ``df`` to a Modin DataFrame for parallelised processing.
    redis_client : Any, optional
        Redis client used to store and retrieve cached features.
    cache_key : str, optional
        Key under which features are cached in Redis.
    cache_ttl : int, optional
        Override TTL in seconds for the cached features.
    use_dask : bool, optional
        Use Dask to parallelise feature generation.

    Returns
    -------
    pd.DataFrame
        Data frame sorted by ``ts`` with additional feature columns:
        ``ema_short``, ``ema_long``, ``macd`` and parameterized columns
        for RSI, volatility, ATR, Bollinger Bands, momentum, ADX and OBV.
        Missing values are forward-filled and any remaining ``NaN`` rows
        dropped.
    """

    if redis_client is not None and cache_key:
        cached = load_cached_features(redis_client, cache_key)
        if cached is not None:
            return cached

    orig_use_gpu = use_gpu
    use_gpu = use_gpu and has_rocm()
    if orig_use_gpu and not use_gpu:
        logger.info("ROCm not detected; using CPU for features.")

    if "ts" not in df.columns or "price" not in df.columns:
        raise ValueError("DataFrame must contain 'ts' and 'price' columns")

    if use_modin:
        import modin.pandas as mpd  # type: ignore
        backend_df = mpd.DataFrame(df)
    else:
        backend_df = df

    needs_target = "target" not in df.columns or df.get("target", pd.Series()).isna().any()
    warn_overwrite = "target" in df.columns and needs_target

    if use_dask:
        import dask.dataframe as dd  # type: ignore

        def process(pdf: pd.DataFrame) -> pd.DataFrame:
            result, _, _, _ = _compute_features_pandas(
                pdf,
                ema_short_period,
                ema_long_period,
                rsi_period,
                volatility_window,
                atr_window,
                bollinger_window,
                bollinger_std,
                momentum_period,
                adx_period,
                use_numba=use_gpu,
            )
            return result

        ddf = dd.from_pandas(backend_df, npartitions=2)
        backend_df = ddf.map_partitions(process).compute()
        rsi_col = f"rsi{rsi_period}"
        vol_col = f"volatility{volatility_window}"
        atr_col = f"atr{atr_window}"
    else:
        backend_df, rsi_col, vol_col, atr_col = _compute_features_pandas(
            backend_df,
            ema_short_period,
            ema_long_period,
            rsi_period,
            volatility_window,
            atr_window,
            bollinger_window,
            bollinger_std,
            momentum_period,
            adx_period,
            use_numba=use_gpu,
        )

    if use_gpu:
        if jnp is None:
            raise ValueError("jax is required for GPU acceleration")
        _ = jnp.asarray(backend_df.select_dtypes(include=[np.number]).to_numpy())

    if backend_df[[rsi_col, vol_col, atr_col]].isna().all().all():
        raise ValueError("Too many NaN values after interpolation")

    backend_df = backend_df.bfill().ffill()

    if warn_overwrite:
        logger.warning("Overwriting existing target column")
    if needs_target:
        backend_df["target"] = np.sign(backend_df["log_ret"].shift(-1)).fillna(0).astype(int)

    result = backend_df.dropna()
    if needs_target and "price" in result.columns:
        returns = result["price"].pct_change().shift(-1)
        result["target"] = np.where(
            returns > return_threshold,
            1,
            np.where(returns < -return_threshold, -1, 0),
        )
        result["target"] = pd.Series(result["target"], index=result.index).fillna(0)

    if use_modin:
        result = result.to_pandas() if hasattr(result, "to_pandas") else pd.DataFrame(result)

    if redis_client is not None and cache_key:
        store_cached_features(redis_client, cache_key, result, cache_ttl)

    return result
