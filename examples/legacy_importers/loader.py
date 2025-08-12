"""Async data loading utilities for Supabase-backed datasets."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    redis = None

import httpx
import pandas as pd

try:
    from supabase import AuthApiError, Client, create_client
except ImportError:  # pragma: no cover - fallback for older package
    from supabase import Client, create_client
    AuthApiError = Exception

logger = logging.getLogger(__name__)
from tenacity import retry, stop_after_attempt, wait_exponential

from cache_utils import load_cached_features
from utils.normalise import normalize_ohlc

_REDIS_CLIENT = None


def _get_redis_client():
    """Return a configured Redis client or ``None`` if unavailable."""
    global _REDIS_CLIENT
    if redis is None:
        logger.warning("Redis package not installed; caching disabled")
        return None
    if os.environ.get("DISABLE_REDIS"):
        return None
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL")
    if url:
        _REDIS_CLIENT = redis.from_url(url)
        return _REDIS_CLIENT

    if os.getenv("REDIS_HOST") or os.getenv("REDIS_PORT") or os.getenv("REDIS_DB"):
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        db = int(os.getenv("REDIS_DB", 0))
        _REDIS_CLIENT = redis.Redis(host=host, port=port, db=db)
        return _REDIS_CLIENT

    return None


def _get_client() -> Client:
    """Create a Supabase client from environment variables.

    The function always initialises the client using the public (anon) key.
    If a JWT or user credentials are provided, it authenticates the client
    accordingly. ``SUPABASE_SERVICE_KEY`` is intentionally ignored here and
    should only be used for write operations such as model uploads.
    """

    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_KEY")
    if not url or not anon_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
        )

    client = create_client(url, anon_key)

    jwt = os.getenv("SUPABASE_JWT")
    email = os.getenv("SUPABASE_USER_EMAIL")
    password = os.getenv("SUPABASE_PASSWORD")

    try:
        if jwt:
            client.auth.set_session(jwt, "")
        elif email and password:
            client.auth.sign_in_with_password({"email": email, "password": password})
    except (httpx.HTTPError, AuthApiError, ValueError) as exc:  # pragma: no cover - requires real Supabase instance
        logger.error("Supabase auth failed: %s", exc)
        raise

    return client


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def _fetch_logs(
    client: Client,
    start_ts: datetime,
    end_ts: datetime,
    *,
    symbol: str | None = None,
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
    redis_client: Any | None,
    cache_key: str | None,
    cache_features: bool,
    feature_params: dict | None,
) -> pd.DataFrame:
    """Compute features and optionally cache them in Redis."""
    from cointrainer.features.build import make_features

    if os.environ.get("DISABLE_FEATURES"):
        return df
    params = feature_params or {}
    if cache_features and redis_client is not None and cache_key is not None:
        feat_key = f"features_{cache_key}"
        return make_features(
            df, redis_client=redis_client, cache_key=feat_key, **params
        )
    return make_features(df, **params)


def _resample_trade_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` sorted by ``ts`` and resampled to 1 minute."""
    if "ts" not in df.columns or df.empty:
        return df

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.sort_values("ts")
    df = df.set_index("ts")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    df_resampled = df.resample("1T").asfreq()

    if numeric_cols:
        df_resampled[numeric_cols] = df_resampled[numeric_cols].interpolate(method="time")

    df_resampled = df_resampled.ffill()

    return df_resampled.reset_index()


def fetch_trade_logs(
    start_ts: datetime,
    end_ts: datetime,
    *,
    symbol: str | None = None,
    table: str = "ohlc_data",
    cache_path: str | None = None,
    redis_client: Any | None = None,
    redis_key: str | None = None,
    cache_only: bool = False,
    max_rows: int | None = None,
    cache_features: bool = False,
    feature_params: dict | None = None,
) -> pd.DataFrame:
    """Return trade logs between ``start_ts`` and ``end_ts`` as a DataFrame.

    When ``cache_only`` is ``True`` and a cached result exists in Redis, the
    cached data is returned without querying the database. Otherwise, the data
    is fetched and the cache is refreshed. If ``max_rows`` is provided the
    DataFrame is truncated to that many rows before any caching occurs.
    """

    if cache_path and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return _resample_trade_logs(df)
    if redis_client is not None:
        key = redis_key or f"{table}:{start_ts.isoformat()}:{end_ts.isoformat()}:{symbol or 'all'}"
        if max_rows is not None:
            key = f"{key}:{max_rows}"
        cached = redis_client.get(key)
        if cached:
            if isinstance(cached, (bytes, bytearray)):
                try:
                    df = pd.read_parquet(BytesIO(cached))
                except Exception:
                    cached = cached.decode()
                    df = pd.read_json(cached, orient="split")
            else:
                df = pd.read_json(cached, orient="split")
            return _resample_trade_logs(df)

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
        if max_rows is not None:
            cache_key = f"{cache_key}_{max_rows}"
        cached = redis_cache.get(cache_key)
        if cache_only and cached:
            df_cached = pd.read_parquet(BytesIO(cached))
            df_cached = _resample_trade_logs(df_cached)
            df_cached = normalize_ohlc(df_cached)
            return _maybe_cache_features(
                df_cached,
                redis_cache,
                cache_key,
                cache_features,
                feature_params,
            )
        if cache_features:
            feat_key = f"features_{cache_key}"
            cached_feat = load_cached_features(redis_cache, feat_key)
            if cached_feat is not None:
                return cached_feat

    if cache_path and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df = _resample_trade_logs(df)
        df = normalize_ohlc(df)
        return _maybe_cache_features(df, redis_cache, cache_key, cache_features, feature_params)

    if redis_client is not None:
        key = (
            redis_key
            or f"{table}:{start_ts.isoformat()}:{end_ts.isoformat()}:{symbol or 'all'}"
        )
        cached = redis_client.get(key)
        if cached:
            if isinstance(cached, (bytes, bytearray)):
                try:
                    df = pd.read_parquet(BytesIO(cached))
                except Exception:
                    cached = cached.decode()
                    df = pd.read_json(cached, orient="split")
            else:
                df = pd.read_json(cached, orient="split")
            df = _resample_trade_logs(df)
            df = normalize_ohlc(df)
            return _maybe_cache_features(
                df, redis_cache, cache_key, cache_features, feature_params
            )
    if redis_cache is not None and cache_only:
        cached = redis_cache.get(cache_key) if cache_key else None
        if cached:
            df = pd.read_parquet(BytesIO(cached))
            df = _resample_trade_logs(df)
            df = normalize_ohlc(df)
            return _maybe_cache_features(
                df, redis_cache, cache_key, cache_features, feature_params
            )

    client = _get_client()

    rows = _fetch_logs(client, start_ts, end_ts, symbol=symbol, table=table)
    df = pd.DataFrame(rows)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            # leave column unchanged if conversion fails
            pass

    if max_rows is not None:
        df = df.head(max_rows)

    df = _resample_trade_logs(df)

    if cache_path:
        df.to_parquet(cache_path)
    if redis_client is not None:
        key = redis_key or f"{table}:{start_ts.isoformat()}:{end_ts.isoformat()}:{symbol or 'all'}"
        if max_rows is not None:
            key = f"{key}:{max_rows}"
        ttl = int(os.getenv("REDIS_TTL", 86400))
        buf = BytesIO()
        df.to_parquet(buf)
        redis_client.setex(key, ttl, buf.getvalue())
        redis_client.setex(key, ttl, df.to_json(orient="split"))

    if redis_cache is not None and cache_key is not None:
        ttl = int(os.getenv("REDIS_TTL", 86400))
        buf = BytesIO()
        df.to_parquet(buf)
        redis_cache.setex(cache_key, ttl, buf.getvalue())
    df = normalize_ohlc(df)
    return _maybe_cache_features(df, redis_cache, cache_key, cache_features, feature_params)


def fetch_trade_aggregates(
    start_ts: datetime,
    end_ts: datetime,
    *,
    symbol: str | None = None,
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
    start_ts: str | None = None,
    end_ts: str | None = None,
    *,
    chunk_size: int = 1000,
    page_size: int | None = None,
    params: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch all rows from ``table`` asynchronously in pages."""

    if start_ts is not None and end_ts is not None:
        return await fetch_data_range_async(table, start_ts, end_ts, chunk_size)

    if page_size is None:
        page_size = 1000

    own_client = False
    if client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
            )
        jwt = os.getenv("SUPABASE_JWT")
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
    params: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
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

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set"
        )

    jwt = os.getenv("SUPABASE_JWT")

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
