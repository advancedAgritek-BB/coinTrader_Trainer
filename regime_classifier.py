"""Utilities for loading a regime classification model and making predictions."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Optional

import joblib
import pandas as pd

from config import load_config

try:  # pragma: no cover - optional dependency
    from supabase import SupabaseException, create_client
except Exception:  # pragma: no cover - supabase not installed
    SupabaseException = Exception  # type: ignore[misc,assignment]
    create_client = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import httpx
except Exception:  # pragma: no cover - httpx not installed
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_MODEL: Optional[object] = None


def load_model() -> object:
    """Return the LightGBM regime model.

    Attempts to download ``regime_lgbm.pkl`` from Supabase Storage using
    credentials provided via :func:`config.load_config`. If the download
    fails for any reason, a base64-encoded fallback model embedded in
    ``train_fallback_model.py`` is used instead.
    """

    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # Try fetching the model from Supabase if possible
    cfg = load_config(require_supabase=False)
    url = cfg.supabase_url
    key = cfg.supabase_key or cfg.supabase_service_key
    if url and key and create_client is not None:
        try:
            client = create_client(url, key)
            data = client.storage.from_("models").download("regime_lgbm.pkl")
            if hasattr(data, "read"):
                data_bytes = data.read()
            else:
                data_bytes = data  # type: ignore[assignment]
            _MODEL = joblib.load(BytesIO(data_bytes))
            return _MODEL
        except Exception as exc:  # pragma: no cover - network/runtime issues
            if httpx is not None and isinstance(exc, httpx.HTTPError):
                logger.warning("Supabase download failed: %s", exc)
            elif isinstance(exc, SupabaseException):
                logger.warning("Supabase error: %s", exc)
            else:
                logger.warning("Model download failed: %s", exc)

    # Fallback to embedded model
    try:
        from train_fallback_model import FALLBACK_MODEL_B64
    except Exception as exc:  # pragma: no cover - missing fallback
        raise RuntimeError("Fallback model missing") from exc

    data = base64.b64decode(FALLBACK_MODEL_B64)
    _MODEL = joblib.load(BytesIO(data))
    return _MODEL


def predict(df: pd.DataFrame) -> pd.Series:
    """Return regime predictions for ``df`` using the loaded model."""

    model = load_model()
    preds = model.predict(df)  # type: ignore[attr-defined]
    # Some models (like LightGBM multiclass) return 2D arrays
    try:
        import numpy as np

        if isinstance(preds, np.ndarray) and preds.ndim > 1:
            preds = preds.argmax(axis=1)
    except Exception:  # pragma: no cover - numpy not installed
        pass

    return pd.Series(preds, index=df.index)
