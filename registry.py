from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
from jsonschema import validate
from supabase import Client, create_client
from tenacity import retry, wait_exponential, stop_after_attempt


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


# Require at least one numeric metric
METRICS_SCHEMA = {
    "type": "object",
    "patternProperties": {"^.+$": {"type": "number"}},
    "minProperties": 1,
    "additionalProperties": {"type": "number"},
}


class ModelRegistry:
    """Registry for ML models stored in Supabase."""

    def __init__(self, url: str, key: str, bucket: str = "models", table: str = "models") -> None:
        self.supabase: Client = create_client(url, key)
        self.bucket = bucket
        self.table = table

    def _hash_bytes(self, data: bytes) -> str:
        """Return the SHA256 hex digest for ``data``."""
        return hashlib.sha256(data).hexdigest()

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
    def upload_bytes(self, payload: bytes, name: str, metrics: Dict[str, Any]) -> ModelEntry:
        """Upload raw ``payload`` bytes as a model artifact."""
        validate(instance=metrics, schema=METRICS_SCHEMA)
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
        data = self.supabase.table(self.table).insert(row).execute().data[0]
        return ModelEntry(**data)

    def upload(self, model_obj: Any, name: str, metrics: Dict[str, Any]) -> ModelEntry:
        """Serialize and upload ``model_obj`` with ``metrics``."""
        if not isinstance(metrics, dict) or not all(isinstance(v, (int, float)) for v in metrics.values()):
            raise ValueError("metrics must be a dict of numeric values")
        validate(instance=metrics, schema=METRICS_SCHEMA)

        buffer = io.BytesIO()
        joblib.dump(model_obj, buffer)
        payload = buffer.getvalue()
        digest = self._hash_bytes(payload)
        path = f"{name}/{digest}.pkl"
        self.supabase.storage.from_(self.bucket).upload(path, io.BytesIO(payload))
        row = {
            "name": name,
            "file_path": path,
            "sha256": digest,
            "metrics": metrics,
            "approved": False,
        }
        data = self.supabase.table(self.table).insert(row).execute().data[0]
        return ModelEntry(**data)

    def upload_dict(self, payload_dict: Dict[str, Any], name: str, metrics: Dict[str, Any]) -> ModelEntry:
        """Serialize and upload a dictionary as JSON."""
        validate(instance=metrics, schema=METRICS_SCHEMA)
        raw = json.dumps(payload_dict).encode()
        return self.upload_bytes(raw, name, metrics)

    def get_latest(self, name: str, approved: bool = True) -> Optional[Tuple[Any, ModelEntry]]:
        """Return the most recent model matching ``name``."""
        query = self.supabase.table(self.table).select("*").eq("name", name)
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
        self.supabase.table(self.table).update({"approved": True}).eq("id", model_id).execute()

    def list_models(self, *, tag: Optional[str] = None, approved: Optional[bool] = None) -> list[ModelEntry]:
        """Return models filtered by tag and approval state."""
        query = self.supabase.table(self.table).select("*")
        if approved is not None:
            query = query.eq("approved", approved)
        if tag is not None:
            query = query.contains("tags", [tag])
        res = query.execute()
        return [ModelEntry(**row) for row in res.data]

