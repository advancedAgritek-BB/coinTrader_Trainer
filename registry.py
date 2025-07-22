"""Utilities to upload and manage models in Supabase Storage."""

from __future__ import annotations

import hashlib
import io
import pickle
import joblib
import joblib
from tenacity import retry, wait_exponential, stop_after_attempt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from jsonschema import validate

from supabase import Client, create_client


@dataclass
class ModelEntry:
    """Representation of a row in the ``models`` table."""

    id: int
    name: str
    file_path: str
    sha256: str
    metrics: Dict[str, Any]
    approved: bool
    tags: Optional[dict] = None


METRICS_SCHEMA = {
    "type": "object",
    "properties": {
        "sharpe": {"type": "number"},
    },
    "required": ["sharpe"],
}


class ModelRegistry:
    """Registry for ML models backed by Supabase."""

    def __init__(self, url: str, key: str, bucket: str = "models") -> None:
        self.supabase: Client = create_client(url, key)
        self.bucket = bucket

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5)
    )
    def upload_bytes(
        self, payload: bytes, name: str, metrics: Dict[str, Any]
    ) -> ModelEntry:
        """Upload raw ``payload`` bytes as a model artifact.

        Parameters
        ----------
        payload:
            Serialized model bytes.
        name:
            Logical model family name.
        metrics:
            Dictionary of evaluation metrics.
        """
        digest = self._hash_bytes(payload)
        path = f"{name}/{digest}.bin"
        self.supabase.storage.from_(self.bucket).upload(path, io.BytesIO(payload))
        row = {
            "name": name,
            "file_path": path,
            "sha256": digest,
            "metrics": metrics,
            "approved": False,
        }
        data = self.supabase.table("models").insert(row).execute().data[0]
        return ModelEntry(**data)

    def upload(
        self,
        model_obj: Any,
        name: str,
        metrics: Dict[str, Any],
        tags: Optional[dict] = None,
    ) -> ModelEntry:
        """Serialize and upload ``model_obj``.

        Parameters
        ----------
        model_obj:
            Arbitrary Python object representing the model.
        name:
            Logical name for the model family.
        metrics:
            Dictionary of evaluation metrics.
        tags:
            Optional dictionary of metadata tags.
        """
        if not isinstance(metrics, dict) or not all(
            isinstance(v, (int, float)) for v in metrics.values()
        ):
            raise ValueError("metrics must be a dict of numeric values")

        buffer = io.BytesIO()
        joblib.dump(model_obj, buffer)
        data_bytes = buffer.getvalue()
        digest = self._hash_bytes(data_bytes)
        buffer = io.BytesIO()
        joblib.dump(model_obj, buffer)
        payload = buffer.getvalue()
        validate(instance=metrics, schema=METRICS_SCHEMA)

        payload = pickle.dumps(model_obj)
        digest = self._hash_bytes(payload)
        path = f"{name}/{digest}.pkl"

        # Upload bytes to Storage
        self.supabase.storage.from_(self.bucket).upload(path, io.BytesIO(data_bytes))

        # Insert metadata row
        row = {
            "name": name,
            "file_path": path,
            "sha256": digest,
            "metrics": metrics,
            "approved": False,
            "tags": tags or {},
        }
        data = self.supabase.table("models").insert(row).execute().data[0]
        return ModelEntry(**data)

    def get_latest(
        self, name: str, approved: bool = True
    ) -> Optional[Tuple[Any, ModelEntry]]:
        """Return the most recent model and its metadata."""
        query = self.supabase.table("models").select("*").eq("name", name)
        if approved is not None:
            query = query.eq("approved", approved)
        res = query.order("created_at", desc=True).limit(1).execute()
        if not res.data:
            return None
        row = ModelEntry(**res.data[0])
        file_bytes = self.supabase.storage.from_(self.bucket).download(row.file_path)
        model = joblib.load(io.BytesIO(file_bytes))
        return model, row

    def approve(self, model_id: int) -> None:
        """Mark a model row as approved."""
        self.supabase.table("models").update({"approved": True}).eq(
            "id", model_id
        ).execute()

    def list_models(
        self, *, tag: Optional[str] = None, approved: Optional[bool] = None
    ) -> list[ModelEntry]:
        """Return models optionally filtered by tag and approval."""
        query = self.supabase.table("models").select("*")
        if approved is not None:
            query = query.eq("approved", approved)
        if tag is not None:
            query = query.contains("tags", [tag])
        res = query.execute()

    def list_models(
        self, name: str, tag_filter: Optional[dict] = None
    ) -> list[ModelEntry]:
        """Return all models matching ``name`` and optional tag filters.

        Parameters
        ----------
        name : str
            Model family name to filter on.
        tag_filter : dict, optional
            Mapping of JSON tag keys to desired values. Each key/value pair
            is added to the query using the ``tags->`` syntax.

        Returns
        -------
        list[ModelEntry]
            List of ``ModelEntry`` objects ordered by newest first.
        """

        query = self.supabase.table("models").select("*").eq("name", name)
        if tag_filter:
            for key, value in tag_filter.items():
                query = query.eq(f"tags->>{key}", value)
        res = query.order("created_at", desc=True).execute()
        return [ModelEntry(**row) for row in res.data]
