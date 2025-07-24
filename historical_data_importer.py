"""Utilities for importing historical OHLCV data into Supabase."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import os

import pandas as pd
from supabase import Client, create_client


def download_historical_data(
    csv_path: str,
    start_ts: Optional[str | datetime] = None,
    end_ts: Optional[str | datetime] = None,
) -> pd.DataFrame:
    """Load ``csv_path`` and return a filtered DataFrame with targets."""
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"close": "price"})

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        raise ValueError("CSV must contain a 'timestamp' or 'ts' column")

    if start_ts is not None:
        start = pd.to_datetime(start_ts)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        df = df[df["ts"] >= start]
    if end_ts is not None:
        end = pd.to_datetime(end_ts)
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        df = df[df["ts"] < end]

    if "price" not in df.columns:
        raise ValueError("CSV must contain a 'close' or 'price' column")

    if "target" not in df.columns:
        df["target"] = (df["price"].shift(-1) > df["price"]).astype(int).fillna(0)

    return df


def _get_write_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


def insert_to_supabase(
    df: pd.DataFrame,
    table: str,
    *,
    client: Optional[Client] = None,
    batch_size: int = 500,
) -> None:
    """Insert ``df`` rows into ``table`` in batches."""
    if client is None:
        client = _get_write_client()

    records = df.to_dict(orient="records")
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        client.table(table).insert(batch).execute()

