"""Lightweight Supabase client helpers with lazy imports."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd


def get_client():
    """Return a configured Supabase client.

    The Supabase SDK is imported lazily to avoid requiring it at runtime if
    the client is never used.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    from supabase import create_client  # type: ignore

    return create_client(url, key)


def select_range(
    table: str,
    start_ts: datetime | str,
    end_ts: datetime | str,
    *,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch rows from ``table`` between ``start_ts`` and ``end_ts``."""
    client = get_client()
    start = start_ts if isinstance(start_ts, str) else start_ts.isoformat()
    end = end_ts if isinstance(end_ts, str) else end_ts.isoformat()
    query = (
        client.table(table).select("*").gte("timestamp", start).lt("timestamp", end)
    )
    if symbol is not None:
        query = query.eq("symbol", symbol)
    data = query.execute().data
    return pd.DataFrame(data)
