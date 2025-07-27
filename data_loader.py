"""Async data loading utilities for Supabase-backed datasets."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, Optional

from dotenv import load_dotenv

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

import httpx
import pandas as pd
from supabase import Client, create_client
from tenacity import retry, stop_after_attempt, wait_exponential
from feature_engineering import make_features

load_dotenv()


_REDIS_CLIENT = None


def _get_redis_client():
    """Return a configured Redis client or ``None`` if unavailable."""
    global _REDIS_CLIENT
    if redis is None:
        return None
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    host = os.environ.get("REDIS_HOST")
    if not host:
        return None
    port = int(os.environ.get("REDIS_PORT", 6379))
    db = int(os.environ.get("REDIS_DB", 0))
    _REDIS_CLIENT = redis.Redis(host=host, port=port, db=db)
    return _REDIS_CLIENT


def _get_client() -> Client:
    """Create a Supabase client from environment variables.

    The function always initialises the client using the public (anon) key.
    If a JWT or user credentials are provided, it authenticates the client
    accordingly. ``SUPABASE_SERVICE_KEY`` is intentionally ignored here and
    should only be used for write operations such as model uploads.
    """

    url = os.environ.get("SUPABASE_URL")
    anon_key = os.environ.get("SUPABASE_KEY")
    if not url or not anon_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
        )

    client = create_client(url, anon_key)

    jwt = os.environ.get("SUPABASE_JWT")
    email = os.environ.get("SUPABASE_USER_EMAIL")
    password = os.environ.get("SUPABASE_PASSWORD")

    try:
        if jwt:
            client.auth.set_session(jwt, "")
        elif email and password:
            client.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as exc:  # pragma: no cover - requires real Supabase instance
        raise RuntimeError("Supabase authentication failed") from exc

    return client


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def _fetch_logs(
    client: Client,
    start_ts: datetime,
    end_ts: datetime,
    *,
    symbol: Optional[str] = None,
    table: str = "ohlc_data",
) -> list[dict]:
    """Fetch rows from ``table`` between ``start_ts`` and ``end_ts`` with retry."""
    query = (
        client.table(table)
        .select("*")
        .gte("timestamp", start_ts.isoformat())
        .lt("timestamp", end_ts.isoformat())
    )
    if symbol is not None:
        query = query.eq("symbol", symbol)
    resp = query.execute()
    return resp.data


def _maybe_cache_features(
    df: pd.DataFrame,
    redis_client: Optional[Any],
    cache_key: Optional[str],
    cache_features: bool,
    feature_params: Optional[dict],
) -> pd.DataFrame:
    """Return ``df`` or cached features depending on ``cache_features``."""
    if not cache_features or redis_client is None or cache_key is None:
        return df
    feat_key = f"features_{cache_key}"
    cached = redis_client.get(feat_key)
    if cached:
        return pd.read_parquet(BytesIO(cached))
    features_df = make_features(df, **(feature_params or {}))
    ttl = int(os.environ.get("REDIS_TTL", 3600))
    buf = BytesIO()
    features_df.to_parquet(buf)
    redis_client.setex(feat_key, ttl, buf.getvalue())
    return features_df


def fetch_trade_logs(
    start_ts: datetime,
    end_ts: datetime,
    *,
    symbol: Optional[str] = None,
    table: str = "ohlc_data",
    cache_path: Optional[str] = None,
    redis_client: Optional[Any] = None,
    redis_key: Optional[str] = None,
    cache_features: bool = False,
    feature_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Return trade logs between ``start_ts`` and ``end_ts`` as a DataFrame."""

    if start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=timezone.utc)
    else:
        start_ts = start_ts.astimezone(timezone.utc)

    if end_ts.tzinfo is None:
        end_ts = end_ts.replace(tzinfo=timezone.utc)
    else:
        end_ts = end_ts.astimezone(timezone.utc)

    redis_cache = _get_redis_client()
    cache_key = None
    if redis_cache is not None:
        cache_key = f"trades_{int(start_ts.timestamp())}_{int(end_ts.timestamp())}_{symbol or ''}"
        if cache_features:
            feat_key = f"features_{cache_key}"
            cached_feat = redis_cache.get(feat_key)
            if cached_feat:
                return pd.read_parquet(BytesIO(cached_feat))

    if cache_path and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return _maybe_cache_features(df, redis_cache, cache_key, cache_features, feature_params)

    if redis_client is not None:
        key = (
            redis_key
            or f"{table}:{start_ts.isoformat()}:{end_ts.isoformat()}:{symbol or 'all'}"
        )
        cached = redis_client.get(key)
        if cached:
            if isinstance(cached, bytes):
                cached = cached.decode()
            df = pd.read_json(cached, orient="split")
            return _maybe_cache_features(df, redis_cache, cache_key, cache_features, feature_params)
    if redis_cache is not None:
        cached = redis_cache.get(cache_key) if cache_key else None
        if cached:
            df = pd.read_parquet(BytesIO(cached))
            return _maybe_cache_features(df, redis_cache, cache_key, cache_features, feature_params)

    client = _get_client()

    rows = _fetch_logs(client, start_ts, end_ts, symbol=symbol, table=table)
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
        key = (
            redis_key
            or f"{table}:{start_ts.isoformat()}:{end_ts.isoformat()}:{symbol or 'all'}"
        )
        redis_client.set(key, df.to_json(orient="split"))

    if redis_cache is not None and cache_key is not None:
        ttl = int(os.environ.get("REDIS_TTL", 3600))
        buf = BytesIO()
        df.to_parquet(buf)
        redis_cache.setex(cache_key, ttl, buf.getvalue())

    return _maybe_cache_features(df, redis_cache, cache_key, cache_features, feature_params)


def fetch_trade_aggregates(
    start_ts: datetime,
    end_ts: datetime,
    *,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Return aggregated trade data between ``start_ts`` and ``end_ts``."""

    client = _get_client()

    if start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=timezone.utc)
    else:
        start_ts = start_ts.astimezone(timezone.utc)

    if end_ts.tzinfo is None:
        end_ts = end_ts.replace(tzinfo=timezone.utc)
    else:
        end_ts = end_ts.astimezone(timezone.utc)

    body = {
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
    }
    if symbol is not None:
        body["symbol"] = symbol

    resp = client.functions.invoke(
        "aggregate-trades", {"body": body, "responseType": "json"}
    )
    data = resp.get("data") if isinstance(resp, dict) and "data" in resp else resp
    df = pd.DataFrame(data)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
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

    jwt = os.environ.get("SUPABASE_JWT")

    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
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
