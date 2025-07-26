"""Feature generation utilities used by coinTrader models."""

import time

import numpy as np
import pandas as pd

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
    high = df["high"]
    low = df["low"]
    close = df["price"].shift()
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def _compute_features_pandas(
    df: pd.DataFrame,
    ema_short_period: int,
    ema_long_period: int,
    rsi_period: int,
    volatility_window: int,
    atr_window: int,
) -> tuple[pd.DataFrame, str, str, str]:
    df = df.sort_values("ts").reset_index(drop=True).copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.set_index("ts").interpolate(method="time").reset_index()
    df = df.ffill()

    df["log_ret"] = np.log(df["price"] / df["price"].shift(1))
    df["ema_short"] = df["price"].ewm(span=ema_short_period, adjust=False).mean()
    df["ema_long"] = df["price"].ewm(span=ema_long_period, adjust=False).mean()
    df["macd"] = df["ema_short"] - df["ema_long"]

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

    return df, rsi_col, vol_col, atr_col


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
        for RSI, volatility and ATR. Missing values are forward-filled and any
        remaining ``NaN`` rows dropped.
    """

    start_time = time.time() if log_time else None

    if "ts" not in df.columns or "price" not in df.columns:
        raise ValueError("DataFrame must contain 'ts' and 'price' columns")

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

            df = gdf.to_pandas() if hasattr(gdf, "to_pandas") else pd.DataFrame(gdf)
        except Exception:
            pdf = gdf.to_pandas() if hasattr(gdf, "to_pandas") else pd.DataFrame(gdf)
            df, rsi_col, vol_col, atr_col = _compute_features_pandas(
                pdf,
                ema_short_period,
                ema_long_period,
                rsi_period,
                volatility_window,
                atr_window,
            )

        if df[[f"rsi{rsi_period}", f"volatility{volatility_window}", f"atr{atr_window}"]].isna().all().all():
            raise ValueError("Too many NaN values after interpolation")

        df = df.bfill().ffill()
        result = df.dropna()
        if "target" not in result.columns and "price" in result.columns:
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
    )

    if df[[rsi_col, vol_col, atr_col]].isna().all().all():
        raise ValueError("Too many NaN values after interpolation")

    df = df.bfill().ffill()
    result = df.dropna()
    if "target" not in result.columns and "price" in result.columns:
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
