"""Utilities for working with DirectML devices."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_dml_device() -> Any:
    """Return a DirectML ``device`` or CPU fallback.

    The function attempts to create a ``torch_directml.device`` instance.
    On success the chosen device is logged at ``INFO`` level.  If DirectML
    is unavailable a warning is logged and a CPU device is returned.  When
    PyTorch itself is not installed, the string ``"cpu"`` is returned.
    """
    try:
        import torch_directml  # type: ignore

        device = torch_directml.device(0)
        logger.info(f"Using DirectML device: {device}")
        return device
    except Exception as exc:
        logger.warning("DirectML not available, falling back to CPU: %s", exc)

    try:
        import torch  # type: ignore

        if getattr(getattr(torch, "version", None), "hip", None):
            logger.info("Using ROCm device")
            return torch.device("cuda")

        return torch.device("cpu")
    except Exception:  # pragma: no cover - PyTorch missing
        logger.warning("PyTorch not installed, returning 'cpu' string")
        return "cpu"
