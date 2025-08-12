"""Optional Redis cache helpers."""
from __future__ import annotations

import os
from typing import Optional

_CACHE = None


def get_cache():
    """Return a Redis client if available, otherwise ``None``.

    The redis package is imported lazily. If the package is missing or a
    connection cannot be established, ``None`` is returned without raising
    an exception.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:  # lazy import
        import redis  # type: ignore
    except Exception:  # pragma: no cover - redis is optional
        return None
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL")
    try:
        if url:
            _CACHE = redis.from_url(url)
        else:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", 6379))
            _CACHE = redis.Redis(host=host, port=port)
        _CACHE.ping()
    except Exception:  # pragma: no cover - cache is optional
        _CACHE = None
    return _CACHE


def get_parquet(key: str) -> Optional[bytes]:
    """Return cached parquet bytes for ``key`` if available."""
    cache = get_cache()
    if cache is None:
        return None
    try:
        return cache.get(key)
    except Exception:  # pragma: no cover - cache failures are ignored
        return None


def set_parquet(key: str, data: bytes) -> None:
    """Store ``data`` under ``key`` if a cache is available."""
    cache = get_cache()
    if cache is None:
        return
    try:
        ttl = int(os.getenv("REDIS_TTL", 86400))
        cache.setex(key, ttl, data)
    except Exception:  # pragma: no cover - cache failures are ignored
        pass
