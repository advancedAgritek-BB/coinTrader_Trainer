from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from supabase import create_client, Client
from tenacity import retry, wait_exponential, stop_after_attempt


def _get_client() -> Client:
    """Create a Supabase client from environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
        )
    return create_client(url, key)


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def _fetch_logs(client: Client, start_ts: datetime, end_ts: datetime) -> list[dict]:
    """Fetch rows from the trade_logs table with retry."""
    response = (
        client.table("trade_logs")
        .select("*")
        .gte("timestamp", start_ts.isoformat())
        .lt("timestamp", end_ts.isoformat())
        .execute()
    )
    return response.data


def fetch_trade_logs(start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    """Return trade logs between two timestamps as a DataFrame."""
    client = _get_client()
    rows = _fetch_logs(client, start_ts, end_ts)
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df
