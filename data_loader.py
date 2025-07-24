"""Async data loading utilities for Supabase-backed datasets."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, Dict, AsyncGenerator, Any
import json

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
def _fetch_logs(
    client: Client,
    start_ts: datetime,
    end_ts: datetime,
    *,
    symbol: Optional[str] = None,
) -> list[dict]:
    """Fetch rows from the ``trade_logs`` table with retry."""
    query = (
        client.table("trade_logs")
        .select("*")
        .gte("timestamp", start_ts.isoformat())
        .lt("timestamp", end_ts.isoformat())
    )
    if symbol is not None:
        query = query.eq("symbol", symbol)
    resp = query.execute()
    return resp.data


def fetch_trade_logs(
    start_ts: datetime,
    end_ts: datetime,
    *,
    symbol: Optional[str] = None,
    cache_path: Optional[str] = None,
    redis_client: Optional[Any] = None,
    redis_key: Optional[str] = None,
) -> pd.DataFrame:
    """Return trade logs between ``start_ts`` and ``end_ts`` as a DataFrame."""

    if cache_path and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    if redis_client is not None:
        key = redis_key or f"trade_logs:{start_ts.isoformat()}:{end_ts.isoformat()}:{symbol or 'all'}"
        cached = redis_client.get(key)
        if cached:
            if isinstance(cached, bytes):
                cached = cached.decode()
            return pd.read_json(cached, orient="split")

    client = _get_client()

    if start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=timezone.utc)
    else:
        start_ts = start_ts.astimezone(timezone.utc)

    if end_ts.tzinfo is None:
        end_ts = end_ts.replace(tzinfo=timezone.utc)
    else:
        end_ts = end_ts.astimezone(timezone.utc)

    rows = _fetch_logs(client, start_ts, end_ts, symbol=symbol)
    df = pd.DataFrame(rows)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            # leave column unchanged if conversion fails
            pass

    if cache_path:
        df.to_parquet(cache_path)
    if redis_client is not None:
        key = redis_key or f"trade_logs:{start_ts.isoformat()}:{end_ts.isoformat()}:{symbol or 'all'}"
        redis_client.set(key, df.to_json(orient="split"))

    return df



async def fetch_table_async(
    table: str,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    *,
    chunk_size: int = 1000,
    page_size: Optional[int] = None,
    params: Optional[Dict[str, str]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> pd.DataFrame:
    """Fetch all rows from ``table`` asynchronously in pages."""

    if start_ts is not None and end_ts is not None:
        return await fetch_data_range_async(table, start_ts, end_ts, chunk_size)

    if page_size is None:
        page_size = 1000

    own_client = False
    if client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
            )
        jwt = os.environ.get("SUPABASE_JWT")
        headers = {"apikey": key, "Authorization": f"Bearer {jwt or key}"}
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
    *,
    page_size: int = 1000,
    params: Optional[Dict[str, str]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> pd.DataFrame:
    """Backward compatible wrapper for ``fetch_table_async``."""
    return await fetch_table_async(
        table,
        start_ts=None,
        end_ts=None,
        chunk_size=1000,
        page_size=page_size,
        params=params,
        client=client,
    )


async def fetch_data_between_async(
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
    """Fetch ``table`` rows between ``start_ts`` and ``end_ts`` asynchronously."""

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
        )

    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
    jwt = os.environ.get("SUPABASE_JWT")
    headers = {"apikey": key, "Authorization": f"Bearer {jwt or key}"}

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
