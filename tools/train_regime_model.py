"""Wrapper to train the regime model and upload via ModelRegistry."""

from __future__ import annotations

from typing import Optional

from registry import ModelRegistry
from train_regime_model import train_regime_model as _train_regime_model


def main(
    data: str,
    *,
    use_gpu: bool = False,
    model_name: str = "regime_model",
    registry: Optional[ModelRegistry] = None,
):
    """Train the regime model from ``data`` and upload it to Supabase."""
    if registry is None:
        registry = ModelRegistry()
    return _train_regime_model(data, use_gpu=use_gpu, model_name=model_name, registry=registry)


__all__ = ["main"]
