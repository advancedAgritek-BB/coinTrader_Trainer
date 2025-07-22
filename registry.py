"""Utilities to upload and manage models in Supabase Storage."""
from __future__ import annotations

import hashlib
import io
import joblib
from tenacity import retry, wait_exponential, stop_after_attempt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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


class ModelRegistry:
    """Registry for ML models backed by Supabase."""

    def __init__(self, url: str, key: str, bucket: str = "models") -> None:
        self.supabase: Client = create_client(url, key)
        self.bucket = bucket

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
    def upload(self, model_obj: Any, name: str, metrics: Dict[str, Any]) -> ModelEntry:
        """Serialize and upload ``model_obj``.

        Parameters
        ----------
        model_obj:
            Arbitrary Python object representing the model.
        name:
            Logical name for the model family.
        metrics:
            Dictionary of evaluation metrics.
        """
        buffer = io.BytesIO()
        joblib.dump(model_obj, buffer)
        payload = buffer.getvalue()
        digest = self._hash_bytes(payload)
        path = f"{name}/{digest}.pkl"

        # Upload bytes to Storage
        self.supabase.storage.from_(self.bucket).upload(path, io.BytesIO(payload))

        # Insert metadata row
        row = {
            "name": name,
            "file_path": path,
            "sha256": digest,
            "metrics": metrics,
            "approved": False,
        }
        data = self.supabase.table("models").insert(row).execute().data[0]
        return ModelEntry(**data)

    def get_latest(self, name: str, approved: bool = True) -> Optional[Tuple[Any, ModelEntry]]:
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
        self.supabase.table("models").update({"approved": True}).eq("id", model_id).execute()
