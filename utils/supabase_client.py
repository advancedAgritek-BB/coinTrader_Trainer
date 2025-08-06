from __future__ import annotations

import base64
import pickle


def download_model(bucket: str, name: str) -> bytes:
    """Download a model from Supabase Storage.

    This minimal stub is provided for testing purposes. In production it should
    be implemented to retrieve ``name`` from ``bucket`` using Supabase.
    """
    raise RuntimeError("Supabase client not implemented")


def load_fallback_model(b64: str):
    """Load a pickled model from a base64-encoded string."""
    data = base64.b64decode(b64)
    return pickle.loads(data)
