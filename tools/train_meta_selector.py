"""Wrapper to train the meta selector model and upload via ModelRegistry."""

from __future__ import annotations

from typing import Optional

from cointrainer.registry import ModelRegistry
from train_meta_selector import train_meta_selector as _train_meta_selector


def main(
    data: str,
    *,
    use_gpu: bool = False,
    model_name: str = "meta_selector",
    registry: Optional[ModelRegistry] = None,
):
    """Train the meta selector from ``data`` and upload it to Supabase."""
    if registry is None:
        registry = ModelRegistry()
    return _train_meta_selector(
        data,
        use_gpu=use_gpu,
        model_name=model_name,
        registry=registry,
    )


__all__ = ["main"]

