from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Optional

from dotenv import load_dotenv

# Load .env once when this module is imported
load_dotenv()


@dataclass
class Config:
    """Holds configuration derived from environment variables."""

    # Default bucket used for storing models in Supabase Storage
    SUPABASE_BUCKET: ClassVar[str] = "models"

    supabase_url: str
    supabase_key: str
    supabase_service_key: Optional[str] = None
    supabase_user_email: Optional[str] = None
    supabase_password: Optional[str] = None
    supabase_jwt: Optional[str] = None
    redis_url: Optional[str] = None
    redis_tls_url: Optional[str] = None
    redis_host: Optional[str] = None
    redis_port: Optional[int] = None
    redis_db: Optional[int] = None
    redis_ttl: Optional[int] = None
    params_bucket: Optional[str] = None
    params_table: Optional[str] = None


def load_config(require_supabase: bool = True) -> Config:
    """Return a :class:`Config` populated from ``os.environ``.

    When ``require_supabase`` is ``False`` only Redis-related fields are
    validated, allowing the caller to read configuration without providing
    Supabase credentials.
    """
    env = os.environ
    url = env.get("SUPABASE_URL")
    key = env.get("SUPABASE_KEY") or env.get("SUPABASE_SERVICE_KEY")
    if require_supabase and (not url or not key):
        missing = []
        if not url:
            missing.append("SUPABASE_URL")
        if not key:
            missing.append("SUPABASE_KEY or SUPABASE_SERVICE_KEY")
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    def _int(name: str) -> Optional[int]:
        val = env.get(name)
        return int(val) if val is not None else None

    return Config(
        supabase_url=url,
        supabase_key=key,
        supabase_service_key=env.get("SUPABASE_SERVICE_KEY"),
        supabase_user_email=env.get("SUPABASE_USER_EMAIL"),
        supabase_password=env.get("SUPABASE_PASSWORD"),
        supabase_jwt=env.get("SUPABASE_JWT"),
        redis_url=env.get("REDIS_URL"),
        redis_tls_url=env.get("REDIS_TLS_URL"),
        redis_host=env.get("REDIS_HOST"),
        redis_port=_int("REDIS_PORT"),
        redis_db=_int("REDIS_DB"),
        redis_ttl=_int("REDIS_TTL"),
        params_bucket=env.get("PARAMS_BUCKET"),
        params_table=env.get("PARAMS_TABLE"),
    )
