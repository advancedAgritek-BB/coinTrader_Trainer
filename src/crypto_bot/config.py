"""Lightweight runtime configuration."""

from __future__ import annotations

import os

from cointrainer import ensure_env_defaults

# Ensure defaults before reading configuration
ensure_env_defaults()


class Config:
    """Runtime configuration populated from environment variables."""

    SYMBOL: str = os.getenv("SYMBOL", "BTCUSDT")
    MODELS_BUCKET: str = os.getenv("CT_MODELS_BUCKET", "models")
    REGIME_PREFIX: str = os.getenv("REGIME_PREFIX", f"{MODELS_BUCKET}/regime/{SYMBOL}")
