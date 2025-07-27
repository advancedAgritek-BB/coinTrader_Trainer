from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import resample

import data_loader


async def prepare_data(
    start_ts: datetime | str | pd.Timestamp,
    end_ts: datetime | str | pd.Timestamp,
    *,
    table: str = "ohlc_data",
    min_rows: int = 1,
    return_threshold: float | None = None,
    symbols: Optional[Iterable[str]] = None,
    use_gpu: bool = False,
    redis_client: Any | None = None,
    cache_key: str | None = None,
    balance: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and targets between ``start_ts`` and ``end_ts``."""

    if isinstance(start_ts, (datetime, pd.Timestamp)):
        start = start_ts.isoformat()
    else:
        start = str(start_ts)
    if isinstance(end_ts, (datetime, pd.Timestamp)):
        end = end_ts.isoformat()
    else:
        end = str(end_ts)

    df = await data_loader.fetch_data_range_async(table, start, end)
    if df.empty:
        logging.error("No data returned for %s - %s", start, end)
        raise ValueError("No data available")

    if len(df) < min_rows:
        raise ValueError(f"Expected at least {min_rows} rows of data, got {len(df)}")

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" not in df.columns:
        try:
            start_dt = pd.to_datetime(start)
        except (TypeError, ValueError):
            start_dt = pd.Timestamp.utcnow()
        df["ts"] = pd.date_range(start_dt, periods=len(df), freq="min")

    if symbols is not None and "symbol" in df.columns:
        df = df[df["symbol"].isin(set(symbols))]

    from feature_engineering import make_features

    loop = asyncio.get_running_loop()
    df = await loop.run_in_executor(
        None,
        lambda: make_features(
            df,
            use_gpu=use_gpu,
            redis_client=redis_client,
            cache_key=cache_key,
        ),
    )

    if "target" not in df.columns:
        if return_threshold is None:
            raise ValueError("Data must contain a 'target' column for training")
        returns = df["price"].pct_change().shift(-1)
        df["target"] = pd.Series(
            np.where(
                returns > return_threshold,
                1,
                np.where(returns < -return_threshold, -1, 0),
            ),
            index=df.index,
        ).fillna(0)

    if balance:
        try:
            counts = df["target"].value_counts()
            max_count = counts.max()
            if len(counts) > 1 and max_count > 0:
                frames = [
                    resample(g, replace=True, n_samples=max_count, random_state=42)
                    for _, g in df.groupby("target")
                ]
                df = (
                    pd.concat(frames)
                    .sample(frac=1.0, random_state=42)
                    .reset_index(drop=True)
                )
        except Exception:
            logging.exception("Failed to balance labels")

    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y
