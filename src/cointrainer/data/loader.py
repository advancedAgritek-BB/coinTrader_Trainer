"""Canonical data loader with optional caching."""
from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from typing import Optional

import pandas as pd

from .cache import get_parquet, set_parquet
from .supabase_client import select_range


def fetch_trade_logs(
    start_ts: datetime | str,
    end_ts: datetime | str,
    symbol: Optional[str],
    *,
    limit: Optional[int] = None,
    cache_path: Optional[str] = None,
    table: str = "trade_logs",
) -> pd.DataFrame:
    """Return trade logs from Supabase or a local cache.

    If ``cache_path`` exists, the Parquet file is read and returned. Otherwise
    the data is fetched from Supabase and optionally written back to
    ``cache_path``. Redis caching is attempted transparently using the helper
    functions in :mod:`cointrainer.data.cache`.
    """
    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    key = f"{table}:{symbol}:{start_ts}:{end_ts}:{limit}"
    cached = get_parquet(key)
    if cached:
        return pd.read_parquet(BytesIO(cached))

    df = select_range(table, start_ts, end_ts, symbol=symbol)
    if limit is not None:
        df = df.head(limit)

    if cache_path:
        df.to_parquet(cache_path)
    buf = BytesIO()
    df.to_parquet(buf)
    set_parquet(key, buf.getvalue())
    return df


async def async_fetch_range(
    table: str,
    start_ts: datetime | str,
    end_ts: datetime | str,
    *,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Asynchronous data fetch stub."""
    raise NotImplementedError("Async fetch not implemented")


# Backwards compatibility
fetch_data_range_async = async_fetch_range
