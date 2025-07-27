"""Feature generation utilities used by coinTrader models."""

import time
import logging

import numpy as np
import pandas as pd

try:
    import cudf  # type: ignore
except Exception:  # pragma: no cover - optional dependency may not be installed
    cudf = None

logger = logging.getLogger(__name__)


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
    df[rsi_col] = _rsi(df["price"], rsi_period)

    vol_col = f"volatility{volatility_window}"
    df[vol_col] = df["log_ret"].rolling(
        window=volatility_window, min_periods=volatility_window
    ).std()

    atr_col = f"atr{atr_window}"
    if {"high", "low"}.issubset(df.columns):
        df[atr_col] = _atr(df, atr_window)
    else:
        df[atr_col] = np.nan

    adx_col = f"adx_{adx_period}"
    if {"high", "low"}.issubset(df.columns):
        df[adx_col] = _adx(df, adx_period)
    else:
        df[adx_col] = np.nan

    df["obv"] = _obv(df) if "volume" in df.columns else np.nan
    # Bollinger Bands (20 period, 2 std dev)
    rolling_mean = df["price"].rolling(window=20, min_periods=20).mean()
    rolling_std = df["price"].rolling(window=20, min_periods=20).std()
    df["bol_mid"] = rolling_mean
    df["bol_upper"] = rolling_mean + 2 * rolling_std
    df["bol_lower"] = rolling_mean - 2 * rolling_std

    # Momentum over 10 periods
    df["momentum_10"] = df["price"] - df["price"].shift(10)

    # ADX indicator
    if {"high", "low"}.issubset(df.columns):
        high = df["high"]
        low = df["low"]
        close = df["price"]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean()
        plus_di = 100 * plus_dm.rolling(window=14, min_periods=14).sum() / atr
        minus_di = 100 * minus_dm.rolling(window=14, min_periods=14).sum() / atr
        dx = 100 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di))
        df["adx_14"] = dx.rolling(window=14, min_periods=14).mean()
    else:
        df["adx_14"] = np.nan

    # On Balance Volume
    vol_col_name = None
    for c in df.columns:
        if c.lower() == "volume" or c.lower().startswith("volume_"):
            vol_col_name = c
            break
    if vol_col_name:
        direction = np.sign(df["price"].diff().fillna(0))
        df["obv"] = (direction * df[vol_col_name]).fillna(0).cumsum()
    else:
        df["obv"] = np.nan

    return df, rsi_col, vol_col, atr_col


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
    log_time: bool = False,
    return_threshold: float = 0.01,
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
    log_time : bool, optional
        Print the elapsed generation time when ``True``.
    return_threshold : float, optional
        Threshold used to generate the ``target`` column when it is missing.

    Returns
    -------
    pd.DataFrame
        Data frame sorted by ``ts`` with additional feature columns:
        ``ema_short``, ``ema_long``, ``macd`` and parameterized columns
        for RSI, volatility, ATR, Bollinger Bands, momentum, ADX and OBV.
        Missing values are forward-filled and any remaining ``NaN`` rows
        dropped.
    """

    start_time = time.time() if log_time else None

    if "ts" not in df.columns or "price" not in df.columns:
        raise ValueError("DataFrame must contain 'ts' and 'price' columns")

    needs_target = "target" not in df.columns or df.get("target", pd.Series()).isna().any()
    warn_overwrite = "target" in df.columns and needs_target

    if use_gpu:
        import cudf as _cudf  # type: ignore

        if _cudf is None:
            raise ValueError("cudf is required for GPU acceleration")

        gdf = _cudf.from_pandas(df)
        # Ensure the cudf DataFrame exposes to_pandas before transformations
        _ = gdf.to_pandas() if hasattr(gdf, "to_pandas") else None

        try:
            gdf = gdf.sort_values("ts").reset_index(drop=True)
            to_dt = getattr(_cudf, "to_datetime", pd.to_datetime)
            gdf["ts"] = to_dt(gdf["ts"], errors="coerce")

            # interpolation is not always implemented in cudf so fall back to pandas
            try:
                gdf = gdf.set_index("ts").interpolate().reset_index()
            except Exception:
                pdf = (
                    gdf.to_pandas() if hasattr(gdf, "to_pandas") else pd.DataFrame(gdf)
                ).set_index("ts").interpolate(method="time").reset_index()
                gdf = _cudf.from_pandas(pdf)

            gdf = gdf.ffill()

            gdf["log_ret"] = np.log(gdf["price"] / gdf["price"].shift(1))
            gdf["ema_short"] = gdf["price"].ewm(span=ema_short_period, adjust=False).mean()
            gdf["ema_long"] = gdf["price"].ewm(span=ema_long_period, adjust=False).mean()
            gdf["macd"] = gdf["ema_short"] - gdf["ema_long"]

            rsi_col = f"rsi{rsi_period}"
            delta = gdf["price"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
            rs = avg_gain / avg_loss
            gdf[rsi_col] = 100 - (100 / (1 + rs))

            vol_col = f"volatility{volatility_window}"
            gdf[vol_col] = gdf["log_ret"].rolling(
                window=volatility_window, min_periods=volatility_window
            ).std()

            atr_col = f"atr{atr_window}"
            if {"high", "low"}.issubset(gdf.columns):
                high = gdf["high"]
                low = gdf["low"]
                close = gdf["price"].shift()
                tr1 = high - low
                tr2 = (high - close).abs()
                tr3 = (low - close).abs()
                tr = _cudf.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                gdf[atr_col] = tr.rolling(window=atr_window, min_periods=atr_window).mean()
            else:
                gdf[atr_col] = np.nan

            # Bollinger Bands
            rolling_mean = gdf["price"].rolling(window=20, min_periods=20).mean()
            rolling_std = gdf["price"].rolling(window=20, min_periods=20).std()
            gdf["bol_mid"] = rolling_mean
            gdf["bol_upper"] = rolling_mean + 2 * rolling_std
            gdf["bol_lower"] = rolling_mean - 2 * rolling_std

            gdf["momentum_10"] = gdf["price"] - gdf["price"].shift(10)

            if {"high", "low"}.issubset(gdf.columns):
                plus_dm = high.diff().clip(lower=0)
                minus_dm = (-low.diff()).clip(lower=0)
                tr1 = high - low
                tr2 = (high - close).abs()
                tr3 = (low - close).abs()
                tr = _cudf.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14, min_periods=14).mean()
                plus_di = 100 * plus_dm.rolling(window=14, min_periods=14).sum() / atr
                minus_di = 100 * minus_dm.rolling(window=14, min_periods=14).sum() / atr
                dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
                gdf["adx_14"] = dx.rolling(window=14, min_periods=14).mean()
            else:
                gdf["adx_14"] = np.nan

            vol_col_name = None
            for c in gdf.columns:
                if c.lower() == "volume" or c.lower().startswith("volume_"):
                    vol_col_name = c
                    break
            if vol_col_name is not None:
                direction = _cudf.Series(np.sign(gdf["price"].diff().fillna(0)))
                gdf["obv"] = (direction * gdf[vol_col_name]).fillna(0).cumsum()
            else:
                gdf["obv"] = np.nan

            df = gdf.to_pandas() if hasattr(gdf, "to_pandas") else pd.DataFrame(gdf)
            mid, upper, lower = _bollinger_bands(df["price"], bollinger_window, bollinger_std)
            df["bol_mid"] = mid
            df["bol_upper"] = upper
            df["bol_lower"] = lower

            mom_col = f"momentum_{momentum_period}"
            df[mom_col] = _momentum(df["price"], momentum_period)

            adx_col = f"adx_{adx_period}"
            if {"high", "low"}.issubset(df.columns):
                df[adx_col] = _adx(df, adx_period)
            else:
                df[adx_col] = np.nan

            df["obv"] = _obv(df) if "volume" in df.columns else np.nan
        except Exception:
            pdf = gdf.to_pandas() if hasattr(gdf, "to_pandas") else pd.DataFrame(gdf)
            df, rsi_col, vol_col, atr_col = _compute_features_pandas(
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
            )

        if df[[f"rsi{rsi_period}", f"volatility{volatility_window}", f"atr{atr_window}"]].isna().all().all():
            raise ValueError("Too many NaN values after interpolation")

        df = df.bfill().ffill()
        if warn_overwrite:
            logger.warning("Overwriting existing target column")
        if needs_target:
            df["target"] = np.sign(df["log_ret"].shift(-1)).fillna(0).astype(int)
        result = df.dropna()
        if needs_target and "price" in result.columns:
            returns = result["price"].pct_change().shift(-1)
            result["target"] = (
                np.where(
                    returns > return_threshold,
                    1,
                    np.where(returns < -return_threshold, -1, 0),
                )
            )
            result["target"] = pd.Series(result["target"], index=result.index).fillna(0)

        if log_time and start_time is not None:
            elapsed = time.time() - start_time
            print(f"feature generation took {elapsed:.3f}s")

        return result

    df, rsi_col, vol_col, atr_col = _compute_features_pandas(
        df,
        ema_short_period,
        ema_long_period,
        rsi_period,
        volatility_window,
        atr_window,
        bollinger_window,
        bollinger_std,
        momentum_period,
        adx_period,
    )

    if df[[rsi_col, vol_col, atr_col]].isna().all().all():
        raise ValueError("Too many NaN values after interpolation")

    df = df.bfill().ffill()
    if warn_overwrite:
        logger.warning("Overwriting existing target column")
    if needs_target:
        df["target"] = np.sign(df["log_ret"].shift(-1)).fillna(0).astype(int)
    result = df.dropna()
    if needs_target and "price" in result.columns:
        returns = result["price"].pct_change().shift(-1)
        result["target"] = (
            np.where(
                returns > return_threshold,
                1,
                np.where(returns < -return_threshold, -1, 0),
            )
        )
        result["target"] = pd.Series(result["target"], index=result.index).fillna(0)

    if log_time and start_time is not None:
        elapsed = time.time() - start_time
        print(f"feature generation took {elapsed:.3f}s")

    return result
