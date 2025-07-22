from __future__ import annotations

import os
from datetime import datetime, timezone
from datetime import datetime
from typing import Optional, Dict, AsyncGenerator
from typing import Optional, Dict
from typing import AsyncGenerator
import pytz

import httpx
import pandas as pd
from supabase import Client, create_client
from tenacity import retry, wait_exponential, stop_after_attempt


def _get_client() -> Client:
    """Create a Supabase client from ``SUPABASE_URL`` and ``SUPABASE_KEY``."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    return create_client(url, key)


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def _fetch_logs(client: Client, start_ts: datetime, end_ts: datetime) -> list[dict]:
    """Fetch rows from ``trade_logs`` with retries."""
    """Fetch rows from the ``trade_logs`` table with retry."""
    response = (
def _fetch_logs(
    client: Client,
    start_ts: datetime,
    end_ts: datetime,
    symbol: str | None = None,
) -> list[dict]:
    """Fetch rows from the ``trade_logs`` table with retry."""

    *,
    symbol: Optional[str] = None,
) -> list[dict]:
    """Fetch rows from the trade_logs table with retry."""
    query = (
        client.table("trade_logs")
        .select("*")
        .gte("timestamp", start_ts.isoformat())
        .lt("timestamp", end_ts.isoformat())
    )

    if symbol is not None:
        query = query.eq("symbol", symbol)

    symbol: Optional[str] = None,
) -> list[dict]:
    """Fetch rows from the trade_logs table with retry."""
    start_ts = start_ts.astimezone(pytz.UTC).isoformat()
    end_ts = end_ts.astimezone(pytz.UTC).isoformat()

    query = (
        client.table("trade_logs")
        .select("*")
        .gte("timestamp", start_ts)
        .lt("timestamp", end_ts)
    )
    if symbol is not None:
        query = query.eq("symbol", symbol)
    response = query.execute()
    return response.data


def fetch_trade_logs(start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    """Return trade logs between ``start_ts`` and ``end_ts`` as a DataFrame."""
    """Return trade logs between two timestamps as a ``DataFrame``."""
def fetch_trade_logs(
    start_ts: datetime,
    end_ts: datetime,
    *,
    symbol: str | None = None,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Return trade logs between ``start_ts`` and ``end_ts`` as a DataFrame.

    Parameters
    ----------
    start_ts, end_ts : datetime
        Timestamp range expressed in UTC.  Naive values are interpreted as
        UTC and converted accordingly.
    symbol : str, optional
        Restrict the returned rows to a specific trading pair.
    cache_path : str, optional
        Location of a Parquet file used as a cache. When provided and the
        file exists, trade logs are loaded from this file instead of
        fetching from Supabase.  Fresh results are written back to this
        path on successful retrieval.

    Returns
    -------
    pd.DataFrame
        DataFrame of trade logs ordered by timestamp.
    """

    client = _get_client()

    if start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=timezone.utc)
    else:
        start_ts = start_ts.astimezone(timezone.utc)

    if end_ts.tzinfo is None:
        end_ts = end_ts.replace(tzinfo=timezone.utc)
    else:
        end_ts = end_ts.astimezone(timezone.utc)

    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    rows = _fetch_logs(client, start_ts, end_ts, symbol)
    symbol: Optional[str] = None,
    cache_file: Optional[str] = None,
) -> pd.DataFrame:
    """Return trade logs between two timestamps as a DataFrame.

    When ``cache_file`` is provided and exists, the Parquet file is loaded
    instead of querying Supabase. Otherwise rows are fetched and optionally
    written to ``cache_file``.
    """

    if cache_file and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    client = _get_client()
    rows = _fetch_logs(client, start_ts, end_ts, symbol=symbol)
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    if cache_path:
        df.to_parquet(cache_path)
    if cache_file:
        df.to_parquet(cache_file)

    return df


async def fetch_table_async(
    cache_path: str = "cache.parquet",
) -> pd.DataFrame:
    """Return trade logs between two timestamps as a DataFrame."""
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    client = _get_client()
    rows = _fetch_logs(client, start_ts, end_ts, symbol)
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    df.to_parquet(cache_path)
    return df


async def fetch_data_async(
    table: str,
    *,
    page_size: int = 1000,
    params: Optional[Dict[str, str]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> pd.DataFrame:
    """Fetch rows from ``table`` asynchronously.

    When ``start_ts`` and ``end_ts`` are provided rows are fetched in ``chunk_size``
    batches between the timestamps. Otherwise the entire table is retrieved in
    pages of ``page_size``.
    """Fetch all rows from ``table`` asynchronously in pages."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    """Fetch all rows from ``table`` asynchronously handling pagination.

    Parameters
    ----------
    table : str
        Table name to query from Supabase REST API.
    page_size : int, optional
        Number of rows per request. Defaults to ``1000``.
    params : dict, optional
        Additional query parameters added to the request. ``select`` defaults
        to ``"*"``.
    client : httpx.AsyncClient, optional
        Client instance preconfigured with base URL and auth headers. When not
        provided one is created from ``SUPABASE_URL`` and ``SUPABASE_KEY``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all retrieved rows.
    """

    own_client = False
    if client is None:
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        client = httpx.AsyncClient(base_url=url, headers=headers)
        own_client = True

    params = params.copy() if params else {}
    params.setdefault("select", "*")

    frames: list[pd.DataFrame] = []
    start = 0

    try:
        while True:
            end = start + page_size - 1
            resp = await client.get(
                f"/rest/v1/{table}",
                params=params,
                headers={"Range-Unit": "items", "Range": f"{start}-{end}"},
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            frames.append(pd.DataFrame(data))
            if len(data) < page_size:
                break
            start += page_size
    finally:
        if own_client:
            await client.aclose()

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


async def fetch_all_rows_async(
    table: str,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    *,
    chunk_size: int = 1000,
    page_size: Optional[int] = None,
    params: Optional[Dict[str, str]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> pd.DataFrame:
    """Backward compatible wrapper for ``fetch_table_async``."""

    return await fetch_table_async(
        table,
        start_ts=start_ts,
        end_ts=end_ts,
        chunk_size=chunk_size,
        page_size=page_size,
        params=params,
        client=client,
    )


async def fetch_data_async(
    table: str,
    start_ts: str,
    end_ts: str,
    *,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """Backward compatible wrapper for fetching rows in a date range."""

    return await fetch_data_range_async(table, start_ts, end_ts, chunk_size)


async def _fetch_chunks(
    client: httpx.AsyncClient,
    endpoint: str,
    start_ts: str,
    end_ts: str,
    chunk_size: int,
) -> AsyncGenerator[pd.DataFrame, None]:
    offset = 0
    while True:
        params = [
            ("select", "*"),
            ("order", "timestamp.asc"),
            ("timestamp", f"gte.{start_ts}"),
            ("timestamp", f"lt.{end_ts}"),
            ("limit", str(chunk_size)),
            ("offset", str(offset)),
        ]
        resp = await client.get(endpoint, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        yield pd.DataFrame(data)
        if len(data) < chunk_size:
            break
        offset += chunk_size


async def fetch_data_range_async(
    table: str,
    start_ts: str,
    end_ts: str,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """Fetch ``table`` rows between ``start_ts`` and ``end_ts`` asynchronously."""

    """Fetch rows between two timestamps in ``chunk_size`` batches."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    chunks: list[pd.DataFrame] = []
    async with httpx.AsyncClient(headers=headers, timeout=None) as client:
        async for chunk in _fetch_chunks(client, endpoint, start_ts, end_ts, chunk_size):
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df
