"""Artifact registry helpers for coinTrader models.

This module provides a small :class:`ModelRegistry` used in legacy code and
lightweight functions :func:`save_model` and :func:`load_latest` that implement
the new artifact contract.  Supabase SDK imports are performed lazily so that
importing :mod:`cointrainer.registry` has no heavy side effects.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pickle
import platform
import tempfile
from datetime import UTC, datetime
from typing import Any

import joblib

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
        url: str | None = None,
        key: str | None = None,
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
        metadata: dict[str, Any] | None = None,
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

        now = datetime.now(UTC).isoformat()
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

    def list_models(self, approved: bool | None = None) -> list[dict[str, Any]]:
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
        metadata: dict | None = None,
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

        now = datetime.now(UTC).isoformat()
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


def _debug(msg: str) -> None:
    if os.getenv("CT_REGISTRY_DEBUG") == "1":
        print(f"[registry] {msg}")


def _get_client() -> Any:  # pragma: no cover - thin wrapper
    """Return a Supabase client using environment variables."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RegistryError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


def _get_bucket() -> str:
    """Return the name of the Supabase storage bucket."""
    bucket = os.getenv("CT_MODELS_BUCKET", "models")
    if not bucket:
        raise RegistryError("CT_MODELS_BUCKET not set")
    return bucket


def _sha256(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def save_model(key: str, blob: bytes, metadata: dict) -> None:
    """Upload model bytes and update ``LATEST.json`` with metadata.

    The model and pointer use ``upsert`` so repeated uploads are idempotent
    and any failure raises :class:`RegistryError`.
    """

    cli = _get_client()
    bucket = _get_bucket()

    file_opts = {"contentType": "application/octet-stream", "upsert": "true"}
    _debug(f"upload {bucket}/{key} (len={len(blob)})")
    res = cli.storage.from_(bucket).upload(key, blob, file_opts)
    if isinstance(res, dict) and res.get("error"):
        raise RegistryError(f"upload error: {res['error']}")

    prefix = key.rsplit("/", 1)[0]
    pointer_path = f"{prefix}/LATEST.json"
    meta = dict(metadata or {})
    meta.setdefault("schema_version", "1")
    meta.setdefault("key", key)
    meta.setdefault("hash", _sha256(blob))
    pointer_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    _debug(f"write pointer -> {bucket}/{pointer_path}")
    cli.storage.from_(bucket).upload(
        pointer_path, pointer_bytes, {"contentType": "application/json", "upsert": "true"}
    )


def load_pointer(prefix: str) -> dict:
    """Read and return the ``LATEST.json`` metadata for ``prefix``."""

    cli = _get_client()
    bucket = _get_bucket()
    pointer = f"{prefix}/LATEST.json"
    _debug(f"download pointer {bucket}/{pointer}")
    try:
        data = cli.storage.from_(bucket).download(pointer)
    except Exception as e:  # pragma: no cover - network errors
        raise RegistryError(f"pointer download failed: {e}") from e
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:  # pragma: no cover - invalid JSON
        raise RegistryError(f"invalid pointer JSON: {e}") from e


def load_latest(prefix: str) -> bytes:
    """Use pointer metadata to download model bytes."""

    meta = load_pointer(prefix)
    key = meta.get("key")
    if not key:
        raise RegistryError("pointer missing 'key'")
    cli = _get_client()
    bucket = _get_bucket()
    _debug(f"download model {bucket}/{key}")
    try:
        return cli.storage.from_(bucket).download(key)
    except Exception as e:  # pragma: no cover - network errors
        raise RegistryError(f"model download failed: {e}") from e


# ---------------------------------------------------------------------------
# Supabase registry publisher ------------------------------------------------
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    """Return SHA256 hex digest for ``data``."""

    return hashlib.sha256(data).hexdigest()


def _now_version() -> str:
    """Return timestamp string ``YYYYMMDD-HHMMSS`` for current UTC time."""

    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _detect_lgbm_version() -> str:
    """Return LightGBM version if installed, otherwise ``unknown``."""

    try:  # pragma: no cover - optional dependency
        import lightgbm as lgb  # type: ignore

        return getattr(lgb, "__version__", "unknown")
    except Exception:  # pragma: no cover - optional dependency
        return "unknown"


class SupabaseRegistry:
    """Publisher for uploading regime models to Supabase Storage."""

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        bucket: str | None = None,
        regime_prefix: str | None = None,
    ) -> None:
        url = url or os.getenv("SUPABASE_URL")
        key = key or os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

        bucket = bucket or os.getenv("CT_MODELS_BUCKET", "models")
        regime_prefix = regime_prefix or os.getenv("CT_REGIME_PREFIX", "models/regime")

        self.client = create_client(url, key)
        self.bucket = bucket
        self.regime_prefix = regime_prefix.rstrip("/")

    def publish_regime_model(
        self,
        model: Any,
        *,
        symbol: str,
        feature_list: list[str],
        label_order: list[int],
        horizon: str,
        thresholds: dict[str, Any],
        code_sha: str,
        data_fingerprint: str,
    ) -> dict:
        """Upload ``model`` and update ``LATEST.json`` pointer for ``symbol``."""

        artifact_bytes = pickle.dumps(model)
        sha256 = _sha256_bytes(artifact_bytes)
        version = _now_version()
        artifact_key = f"{self.regime_prefix}/{symbol}/{version}_regime_lgbm.pkl"
        storage = self.client.storage.from_(self.bucket)
        storage.upload(
            path=artifact_key,
            file=io.BytesIO(artifact_bytes),
            file_options={"content-type": "application/pickle", "upsert": "false"},
        )

        pointer = {
            "key": artifact_key,
            "schema_version": "1",
            "feature_list": feature_list,
            "label_order": label_order,
            "horizon": horizon,
            "thresholds": thresholds,
            "hash": f"sha256:{sha256}",
            "py_version": platform.python_version(),
            "lgbm_version": _detect_lgbm_version(),
            "code_sha": code_sha,
            "data_fingerprint": data_fingerprint,
        }

        latest_key = f"{self.regime_prefix}/{symbol}/LATEST.json"
        storage.upload(
            path=latest_key,
            file=io.BytesIO(json.dumps(pointer).encode()),
            file_options={"content-type": "application/json", "upsert": "true"},
        )

        with contextlib.suppress(Exception):  # optional table insert
            self.client.table("models").upsert(
                {"key": artifact_key, "hash": f"sha256:{sha256}"},
                on_conflict="key",
            ).execute()

        return {
            "version": version,
            "artifact_key": artifact_key,
            "sha256": sha256,
            "latest_key": latest_key,
        }


__all__ = [
    "ModelRegistry",
    "RegistryError",
    "SupabaseRegistry",
    "load_latest",
    "load_pointer",
    "save_model",
]

