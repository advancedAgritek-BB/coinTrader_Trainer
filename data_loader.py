from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Dict
from typing import AsyncGenerator

import httpx

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


async def fetch_all_rows_async(
    table: str,
    *,
    page_size: int = 1000,
    params: Optional[Dict[str, str]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> pd.DataFrame:
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
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
            )
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


async def fetch_data_async(
    table: str,
    start_ts: str,
    end_ts: str,
    *,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """Backward compatible wrapper for ``fetch_data_range_async``."""

    return await fetch_data_range_async(table, start_ts, end_ts, chunk_size)
async def _fetch_chunks(
    client: httpx.AsyncClient,
    endpoint: str,
    start_ts: str,
    end_ts: str,
    chunk_size: int,
) -> AsyncGenerator[pd.DataFrame, None]:
    """Yield DataFrames of rows fetched in chunks from Supabase."""
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
        response = await client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
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
    """Fetch ``table`` rows between ``start_ts`` and ``end_ts`` asynchronously.

    Data are retrieved in ``chunk_size`` batches and concatenated into a single
    ``DataFrame``. Numeric columns are coerced to numbers when possible.
    """

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
        )

    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    chunks: list[pd.DataFrame] = []
    async with httpx.AsyncClient(headers=headers, timeout=None) as client:
        async for chunk in _fetch_chunks(
            client, endpoint, start_ts, end_ts, chunk_size
        ):
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df
