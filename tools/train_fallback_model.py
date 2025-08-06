"""Wrapper to train the fallback model and upload via ModelRegistry."""

from __future__ import annotations

from typing import Optional

from registry import ModelRegistry
from train_fallback_model import train_fallback_model as _train_fallback_model


def main(
    data: str | None = None,
    *,
    model_name: str = "fallback_model",
    registry: Optional[ModelRegistry] = None,
):
    """Train the fallback model and upload its weights to Supabase."""
    payload = _train_fallback_model(data)
    if registry is None:
        registry = ModelRegistry()
    registry.upload_dict({"weights_b64": payload}, model_name)
    return payload


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
