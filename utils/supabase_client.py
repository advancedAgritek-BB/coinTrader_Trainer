from __future__ import annotations

"""Lightweight helpers for interacting with Supabase Storage."""

import base64
import os
import pickle
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables from a .env file if present
load_dotenv()


def get_client() -> Client:
    """Return a Supabase client configured from environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


def upload_model(bucket: str, file_path: str, model_path: str) -> None:
    """Upload a model file to Supabase Storage.

    Parameters
    ----------
    bucket : str
        Name of the storage bucket.
    file_path : str
        Destination path within the bucket.
    model_path : str
        Local filesystem path to the model file.
    """
    client = get_client()
    with open(model_path, "rb") as fh:
        data = fh.read()
    client.storage.from_(bucket).upload(file_path, data)


def download_model(bucket: str, file_path: str) -> bytes:
    """Download a model file from Supabase Storage.

    Parameters
    ----------
    bucket : str
        Name of the storage bucket.
    file_path : str
        Path within the bucket to download.

    Returns
    -------
    bytes
        The downloaded file contents.
    """
    client = get_client()
    data = client.storage.from_(bucket).download(file_path)

    # ``download`` may return a ``bytes`` object or a file-like object
    # depending on the Supabase client version.  Normalise the output to
    # raw bytes so callers have a consistent interface.
    if hasattr(data, "read"):
        return data.read()
    return data


def load_fallback_model(b64_str: str) -> Any:
    """Return a model decoded from a base64 encoded pickle string."""
    raw = base64.b64decode(b64_str.encode("utf-8"))
    return pickle.loads(raw)


__all__ = [
    "get_client",
    "upload_model",
    "download_model",
    "load_fallback_model",
]
