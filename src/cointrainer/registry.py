"""Artifact registry helpers for coinTrader models.

This module provides a small :class:`ModelRegistry` used in legacy code and
lightweight functions :func:`save_model` and :func:`load_latest` that implement
the new artifact contract.  Supabase SDK imports are performed lazily so that
importing :mod:`cointrainer.registry` has no heavy side effects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy registry class -----------------------------------------------------
# ---------------------------------------------------------------------------


def create_client(url: str, key: str) -> Any:  # pragma: no cover - thin wrapper
    """Return a Supabase client by importing the SDK lazily."""

    from supabase import create_client as _create_client  # local import

    return _create_client(url, key)


class ModelRegistry:
    """Registry for storing trained models in Supabase Storage.

    This class is kept for backward compatibility with older training scripts.
    New code should prefer :func:`save_model`.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket: str = "models",
        table: str = "models",
        *,
        client: Any | None = None,
    ) -> None:
        if client is None:
            url = url or os.getenv("SUPABASE_URL")
            key = key or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
            if not url or not key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
            client = create_client(url, key)

        self.client = client
        self.bucket = bucket
        self.table = table

    def upload(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        approved: bool = False,
        conflict_key: str | None = None,
    ) -> str:
        """Upload ``model`` to Supabase Storage under ``name``.

        The model is serialised with :func:`joblib.dump` before upload.  A
        corresponding row is inserted/updated in the ``models`` table with the
        provided metadata.
        """

        path = f"{name}.pkl"
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                joblib.dump(model, tmp.name)
                with open(tmp.name, "rb") as fh:
                    self.client.storage.from_(self.bucket).upload(path, fh.read())
            finally:
                os.unlink(tmp.name)

        now = datetime.utcnow().isoformat()
        entry = {
            "name": name,
            "path": path,
            "metadata": metadata or {},
            "approved": approved,
            "created_at": now,
            "updated_at": now,
        }
        table = self.client.table(self.table)
        params = {"on_conflict": conflict_key} if conflict_key else {}
        resp = table.upsert(entry, **params).execute()
        return resp.data[0]["id"]

    def list_models(self, approved: bool | None = None) -> List[Dict[str, Any]]:
        query = self.client.table(self.table).select("*")
        if approved is not None:
            query = query.eq("approved", approved)
        resp = query.execute()
        return resp.data

    def approve(self, model_id: str) -> None:
        self.client.table(self.table).update({"approved": True}).eq(
            "id", model_id
        ).execute()

    def upload_dict(
        self,
        obj: dict,
        name: str,
        metadata: Optional[dict] = None,
        *,
        approved: bool = False,
        conflict_key: str | None = None,
    ) -> str:
        """Upload a dictionary as JSON and return the inserted row ID."""

        path = f"{name}.json"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w+b") as tmp:
            try:
                tmp.write(json.dumps(obj).encode())
                tmp.flush()
                tmp.seek(0)
                self.client.storage.from_(self.bucket).upload(path, tmp.read())
            finally:
                os.unlink(tmp.name)

        now = datetime.utcnow().isoformat()
        entry = {
            "name": name,
            "path": path,
            "metadata": metadata or {},
            "approved": approved,
            "created_at": now,
            "updated_at": now,
        }
        table = self.client.table(self.table)
        params = {"on_conflict": conflict_key} if conflict_key else {}
        resp = table.upsert(entry, **params).execute()
        return resp.data[0]["id"]


# ---------------------------------------------------------------------------
# New lightweight helpers ---------------------------------------------------
# ---------------------------------------------------------------------------


class RegistryError(Exception):
    pass


def _get_bucket():  # pragma: no cover - thin wrapper
    """Return the Supabase storage bucket handle."""

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RegistryError("SUPABASE_URL and SUPABASE_KEY must be set")

    bucket_name = os.getenv("MODELS_BUCKET", "models")
    client = create_client(url, key)
    return client.storage.from_(bucket_name)


def _upload(path: str, data: bytes) -> None:
    bucket = _get_bucket()
    bucket.upload(path, data, {"upsert": True})


def _download(path: str) -> bytes:
    bucket = _get_bucket()
    return bucket.download(path)


def _move(src: str, dst: str) -> None:
    bucket = _get_bucket()
    bucket.move(src, dst)


def save_model(key: str, blob: bytes, metadata: dict) -> None:
    """Persist a pickled model ``blob`` under ``key`` with ``metadata``.

    The model bytes are uploaded and a ``LATEST.json`` pointer is written
    atomically in the same directory.  ``metadata`` is stored alongside the
    pointer and augmented with a SHA256 hash of ``blob`` when missing.
    """

    prefix = key.rsplit("/", 1)[0]

    meta = dict(metadata)
    meta.setdefault("hash", f"sha256:{hashlib.sha256(blob).hexdigest()}")
    pointer = {"key": key, "schema_version": "1", **meta}

    tmp_name = f"{prefix}/LATEST.{uuid.uuid4().hex}.json"
    try:
        _upload(key, blob)
        _upload(tmp_name, json.dumps(pointer).encode())
        _move(tmp_name, f"{prefix}/LATEST.json")
    except Exception as exc:  # pragma: no cover - network errors
        raise RegistryError(f"failed to save model: {exc}") from exc


def load_pointer(prefix: str) -> dict:
    """Return metadata from ``{prefix}/LATEST.json``."""

    path = f"{prefix}/LATEST.json"
    try:
        data = _download(path)
        return json.loads(data.decode())
    except Exception as exc:
        raise RegistryError(str(exc)) from exc


def load_latest(prefix: str, allow_fallback: bool = False) -> bytes:
    """Return bytes for the model referenced by ``{prefix}/LATEST.json``."""

    pointer_path = f"{prefix}/LATEST.json"
    try:
        info = json.loads(_download(pointer_path).decode())
        key = info["key"]
        blob = _download(key)
        expected = info.get("hash")
        if expected:
            digest = f"sha256:{hashlib.sha256(blob).hexdigest()}"
            if digest != expected:
                raise RegistryError("hash mismatch")
        return blob
    except Exception as exc:
        raise RegistryError(str(exc)) from exc


__all__ = [
    "ModelRegistry",
    "RegistryError",
    "save_model",
    "load_pointer",
    "load_latest",
]

