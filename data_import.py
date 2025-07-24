from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import Optional

import httpx
import pandas as pd
from supabase import create_client

load_dotenv()


def download_historical_data(
    source_url: str,
    symbol: str,
    start_ts: str,
    end_ts: str,
    *,
    batch_size: int = 1000,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Download historical trade data and optionally save to ``output_file``."""
    params = {
        "symbol": symbol,
        "start": start_ts,
        "end": end_ts,
        "limit": batch_size,
    }
    response = httpx.get(source_url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    if output_file:
        df.to_parquet(output_file)
    return df


def insert_to_supabase(
    df: pd.DataFrame,
    *,
    table: str = "trade_logs",
    batch_size: int = 1000,
) -> None:
    """Insert ``df`` rows into ``table`` using Supabase credentials."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and service key must be set")

    client = create_client(url, key)
    records = df.to_dict(orient="records")
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        client.table(table).insert(chunk).execute()

