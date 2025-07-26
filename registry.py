"""Registry for uploading models to Supabase."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for storing trained models in Supabase Storage."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket: str = "models",
        table: str = "models",
        *,
        client: Optional[Client] = None,
    ) -> None:
        """Initialize a model registry client.

        If ``client`` is provided, it is used directly. Otherwise a new client
        is created from ``url`` and ``key`` or environment variables.
        """
        if client is None:
            url = url or os.environ.get("SUPABASE_URL")
            key = key or os.environ.get("SUPABASE_KEY") or os.environ.get(
                "SUPABASE_SERVICE_KEY"
            )
            if not url or not key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
            self.client = create_client(url, key)
        else:
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
    ) -> str:
        """Upload ``model`` to Supabase Storage under ``name`` and return the model ID.

        The ``model`` is serialized using ``joblib.dump`` and uploaded to the
        configured bucket. A record is inserted into the configured table with
        the model metadata and ``approved`` status.
        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            joblib.dump(model, temp_file.name)
            path = f"{name}.pkl"
            with open(temp_file.name, "rb") as f:
                self.client.storage.from_(self.bucket).upload(path, f.read())

        os.unlink(temp_file.name)

        now = datetime.utcnow().isoformat()
        entry = {
            "name": name,
            "path": path,
            "metadata": metadata or {},
            "approved": approved,
            "created_at": now,
            "updated_at": now,
        }
        resp = self.client.table(self.table).insert(entry).execute()
        return resp.data[0]["id"]

    def list_models(self, approved: bool | None = None) -> List[Dict[str, Any]]:
        """List all models in the registry, optionally filtered by ``approved``."""
        query = self.client.table(self.table).select("*")
        if approved is not None:
            query = query.eq("approved", approved)
        resp = query.execute()
        return resp.data

    def approve(self, model_id: str) -> None:
        """Approve the model with ``model_id``."""
        self.client.table(self.table).update({"approved": True}).eq(
            "id", model_id
        ).execute()
