"""Utilities for importing historical OHLCV data into Supabase."""
"""Utilities for importing and uploading historical trading data."""

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

import pandas as pd
from supabase import create_client, Client
from tenacity import retry, wait_exponential, stop_after_attempt


def download_historical_data(
    url: str,
    *,
    symbol: Optional[str] = None,
    start_ts: Optional[datetime | str] = None,
    end_ts: Optional[datetime | str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Return normalized historical price data from ``url``.

    Parameters
    ----------
    url : str
        HTTP or file path to a CSV file.
    symbol : str, optional
        Symbol to filter rows by.
    start_ts : datetime or str, optional
        Inclusive start timestamp for filtering.
    end_ts : datetime or str, optional
        Exclusive end timestamp for filtering.
    output_path : str, optional
        If provided, write the resulting DataFrame to this path. ``.parquet``
        extensions result in Parquet output; otherwise CSV is written.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by ``ts`` with ``target`` column representing the next
        period price increase.
    """

    df = pd.read_csv(url)

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

    df["ts"] = pd.to_datetime(df["ts"])  # type: ignore[arg-type]

    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if start_ts is not None:
        start_ts = pd.to_datetime(start_ts)
        df = df[df["ts"] >= start_ts]
    if end_ts is not None:
        end_ts = pd.to_datetime(end_ts)
        df = df[df["ts"] < end_ts]

    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
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

    url: str,
    key: str,
    *,
    table: str = "historical_prices",
    batch_size: int = 500,
) -> None:
    """Upload ``df`` rows to ``table`` in Supabase.

    Parameters
    ----------
    df : pd.DataFrame
        Data to insert. Columns must match the target table.
    url : str
        Supabase project URL.
    key : str
        API key used to create the client.
    table : str, optional
        Name of the table to insert into. Defaults to ``historical_prices``.
    batch_size : int, optional
        Number of rows per batch insert. Defaults to ``500``.
    """

    client = create_client(url, key)
    records = df.to_dict("records")

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        _insert_batch(client, table, batch)
