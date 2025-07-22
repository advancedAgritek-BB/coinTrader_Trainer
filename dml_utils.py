"""Utilities for working with DirectML devices."""
from __future__ import annotations

from typing import Any
import logging


logger = logging.getLogger(__name__)


def get_dml_device() -> Any:
    """Return a DirectML ``device`` or CPU fallback.

    The function attempts to create a ``torch_directml.device`` instance.
    If the package is not available or any error occurs, a CPU device is
    returned.  When PyTorch itself is not installed, the string ``"cpu"``
    is returned.
    """
    try:
        import torch_directml  # type: ignore

        device = torch_directml.device()
        logger.info("Using DirectML device")
        return device
    except Exception:
        logger.warning("DirectML not available, falling back to CPU")
        try:
            import torch

            return torch.device("cpu")
        except Exception:
            logger.warning("PyTorch not installed, returning 'cpu' string")
            return "cpu"
