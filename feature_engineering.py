"""Feature generation utilities used by coinTrader models."""

import pandas as pd
import numpy as np
import time


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
        If ``True``, compute indicators using PyOpenCL kernels on a
        ``polars.DataFrame`` for GPU acceleration.
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
        import pyopencl as cl
        import polars as pl

        platforms = cl.get_platforms()
        amd = next((p for p in platforms if "AMD" in p.name), None)
        if amd is None:
            raise RuntimeError("No AMD OpenCL platform found")
        ctx = cl.Context(devices=amd.get_devices())
        queue = cl.CommandQueue(ctx)

        df = df.sort_values('ts').reset_index(drop=True).copy()
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
        df = df.set_index('ts').interpolate(method='time').reset_index()
        df = df.ffill()

        pl_df = pl.from_pandas(df)
        price = pl_df['price'].to_numpy().astype(np.float64)
        high = pl_df['high'].to_numpy().astype(np.float64) if 'high' in pl_df.columns else np.zeros_like(price)
        low = pl_df['low'].to_numpy().astype(np.float64) if 'low' in pl_df.columns else np.zeros_like(price)
        n = np.int32(len(price))
        mf = cl.mem_flags
        price_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=price)
        high_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=high)
        low_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=low)
        log_ret = np.zeros_like(price)
        ema_short = np.zeros_like(price)
        ema_long = np.zeros_like(price)
        rsi = np.zeros_like(price)
        volatility = np.zeros_like(price)
        atr = np.zeros_like(price)
        log_ret_buf = cl.Buffer(ctx, mf.WRITE_ONLY, log_ret.nbytes)
        ema_short_buf = cl.Buffer(ctx, mf.WRITE_ONLY, ema_short.nbytes)
        ema_long_buf = cl.Buffer(ctx, mf.WRITE_ONLY, ema_long.nbytes)
        rsi_buf = cl.Buffer(ctx, mf.WRITE_ONLY, rsi.nbytes)
        volatility_buf = cl.Buffer(ctx, mf.WRITE_ONLY, volatility.nbytes)
        atr_buf = cl.Buffer(ctx, mf.WRITE_ONLY, atr.nbytes)

        kernel = """
        #include <math.h>
        __kernel void compute_features(
            __global const double *price,
            __global const double *high,
            __global const double *low,
            const int n,
            const int ema_short_period,
            const int ema_long_period,
            const int rsi_period,
            const int vol_window,
            const int atr_window,
            __global double *log_ret,
            __global double *ema_short,
            __global double *ema_long,
            __global double *rsi,
            __global double *volatility,
            __global double *atr)
        {
            double alpha_s = 2.0 / ((double)ema_short_period + 1.0);
            double alpha_l = 2.0 / ((double)ema_long_period + 1.0);
            double ema_s = price[0];
            double ema_l = price[0];
            log_ret[0] = 0.0;
            ema_short[0] = ema_s;
            ema_long[0] = ema_l;
            for (int i=1; i<n; i++) {
                double p = price[i];
                log_ret[i] = log(p / price[i-1]);
                ema_s += (p - ema_s) * alpha_s;
                ema_l += (p - ema_l) * alpha_l;
                ema_short[i] = ema_s;
                ema_long[i] = ema_l;
            }
            double gain = 0.0;
            double loss = 0.0;
            for (int i=1; i<n; i++) {
                double d = price[i] - price[i-1];
                double up = d > 0 ? d : 0.0;
                double down = d < 0 ? -d : 0.0;
                if (i <= rsi_period) {
                    gain += up;
                    loss += down;
                    if (i == rsi_period) {
                        gain /= rsi_period;
                        loss /= rsi_period;
                        double rs = gain / loss;
                        rsi[i] = 100.0 - 100.0 / (1.0 + rs);
                        for (int j=1; j<rsi_period; j++) rsi[j] = 0.0;
                    }
                } else {
                    gain = (gain * (rsi_period - 1) + up) / rsi_period;
                    loss = (loss * (rsi_period - 1) + down) / rsi_period;
                    double rs = gain / loss;
                    rsi[i] = 100.0 - 100.0 / (1.0 + rs);
                }
            }
            for (int i=0; i<n; i++) {
                if (i + 1 < vol_window) {
                    volatility[i] = 0.0;
                } else {
                    double mean = 0.0;
                    for (int j=i - vol_window + 1; j<=i; j++) {
                        mean += log_ret[j];
                    }
                    mean /= vol_window;
                    double var = 0.0;
                    for (int j=i - vol_window + 1; j<=i; j++) {
                        double diff = log_ret[j] - mean;
                        var += diff * diff;
                    }
                    volatility[i] = sqrt(var / vol_window);
                }
            }
            double prev_atr = 0.0;
            double sum_tr = 0.0;
            for (int i=0; i<n; i++) {
                double prev_close = i==0 ? price[0] : price[i-1];
                double tr1 = high[i] - low[i];
                double tr2 = fabs(high[i] - prev_close);
                double tr3 = fabs(low[i] - prev_close);
                double tr = fmax(tr1, fmax(tr2, tr3));
                if (i < atr_window) {
                    sum_tr += tr;
                    atr[i] = 0.0;
                } else if (i == atr_window) {
                    sum_tr += tr;
                    prev_atr = sum_tr / atr_window;
                    atr[i] = prev_atr;
                } else {
                    prev_atr = (prev_atr * (atr_window - 1) + tr) / atr_window;
                    atr[i] = prev_atr;
                }
            }
        }
        """

        program = cl.Program(ctx, kernel).build()
        program.compute_features(
            queue,
            (1,),
            None,
            price_buf,
            high_buf,
            low_buf,
            n,
            np.int32(ema_short_period),
            np.int32(ema_long_period),
            np.int32(rsi_period),
            np.int32(volatility_window),
            np.int32(atr_window),
            log_ret_buf,
            ema_short_buf,
            ema_long_buf,
            rsi_buf,
            volatility_buf,
            atr_buf,
        )
        cl.enqueue_copy(queue, log_ret, log_ret_buf)
        cl.enqueue_copy(queue, ema_short, ema_short_buf)
        cl.enqueue_copy(queue, ema_long, ema_long_buf)
        cl.enqueue_copy(queue, rsi, rsi_buf)
        cl.enqueue_copy(queue, volatility, volatility_buf)
        cl.enqueue_copy(queue, atr, atr_buf)
        queue.finish()

        pl_df = pl_df.with_columns([
            pl.Series('log_ret', log_ret),
            pl.Series('ema_short', ema_short),
            pl.Series('ema_long', ema_long),
            (pl.Series('ema_short', ema_short) - pl.Series('ema_long', ema_long)).alias('macd'),
            pl.Series(f'rsi{rsi_period}', rsi),
            pl.Series(f'volatility{volatility_window}', volatility),
            pl.Series(f'atr{atr_window}', atr),
        ])

        df = pl_df.to_pandas()
    else:
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
