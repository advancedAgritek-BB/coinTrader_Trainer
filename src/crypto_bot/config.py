"""Lightweight runtime configuration."""

from __future__ import annotations

import os


class Config:
    """Runtime configuration populated from environment variables."""

    SYMBOL: str = os.getenv("SYMBOL", "BTCUSDT")
    MODELS_BUCKET: str = os.getenv("MODELS_BUCKET", "models")
    REGIME_PREFIX: str = os.getenv("REGIME_PREFIX", f"{MODELS_BUCKET}/regime/{SYMBOL}")
