"""Utilities for importing and uploading historical trading data."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import os
from dotenv import load_dotenv

import pandas as pd
from supabase import Client, create_client
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()


def download_historical_data(
    path: str,
    *,
    symbol: Optional[str] = None,
    start_ts: Optional[str | datetime] = None,
    end_ts: Optional[str | datetime] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load historical price data from ``path`` and return a normalized DataFrame."""
    df = pd.read_csv(path)

    rename_map = {
        "timestamp": "ts",
        "close": "price",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "ts" not in df.columns or "price" not in df.columns:
        raise ValueError("CSV must contain timestamp and close/price columns")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if start_ts is not None:
        start_ts = pd.to_datetime(start_ts)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        df = df[df["ts"] >= start_ts]
    if end_ts is not None:
        end_ts = pd.to_datetime(end_ts)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        df = df[df["ts"] < end_ts]

    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    if "target" not in df.columns:
        df["target"] = (df["price"].shift(-1) > df["price"]).fillna(0).astype(int)

    if output_path:
        if output_path.endswith(".parquet"):
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)

    return df


def _get_write_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def _insert_batch(client: Client, table: str, rows: list[dict]) -> None:
    """Insert ``rows`` into ``table`` using ``client`` with retry."""
    client.table(table).insert(rows).execute()


def ensure_table_exists(symbol: str, *, client: Optional[Client] = None) -> str:
    """Create historical prices table for ``symbol`` if needed and return its name."""
    table = f"historical_prices_{symbol.lower()}"
    if client is None:
        client = _get_write_client()
    sql = (
        f"create table if not exists {table} "
        "(like historical_prices including defaults including constraints)"
    )
    # Use Supabase RPC to execute the SQL statement
    client.rpc("sql", {"query": sql}).execute()
    return table


def insert_to_supabase(
    df: pd.DataFrame,
    arg1: str,
    arg2: Optional[str] = None,
    *,
    table: str | None = "historical_prices",
    symbol: Optional[str] = None,
    client: Optional[Client] = None,
    batch_size: int = 500,
) -> None:
    """Insert ``df`` rows into Supabase.

    ``insert_to_supabase(df, table, client=client)`` uses an existing client.
    ``insert_to_supabase(df, url, key, table="tbl")`` creates the client from
    ``url`` and ``key``.
    """
    if arg2 is None:
        table = arg1
        if client is None:
            client = _get_write_client()
    else:
        url = arg1
        key = arg2
        client = create_client(url, key)

    if symbol is not None:
        table = ensure_table_exists(symbol, client=client)
    elif table is None:
        table = "historical_prices"

    records = df.to_dict(orient="records")
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        _insert_batch(client, table, batch)
