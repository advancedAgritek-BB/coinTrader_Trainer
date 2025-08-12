import os
from io import BytesIO
from typing import Any

import pandas as pd

__all__ = ["load_cached_features", "store_cached_features"]


def load_cached_features(redis_client: Any, key: str) -> pd.DataFrame | None:
    """Return cached DataFrame stored under ``key`` or ``None``."""
    if redis_client is None:
        return None
    cached = redis_client.get(key)
    if not cached:
        return None
    return pd.read_parquet(BytesIO(cached))


def store_cached_features(
    redis_client: Any, key: str, df: pd.DataFrame, ttl: int | None = None
) -> None:
    """Store ``df`` in Redis under ``key`` with optional ``ttl``."""
    if redis_client is None:
        return
    ttl = ttl or int(os.environ.get("REDIS_TTL", 86400))
    buf = BytesIO()
    df.to_parquet(buf)
    redis_client.setex(key, ttl, buf.getvalue())
