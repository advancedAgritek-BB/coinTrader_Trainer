"""Utilities for working with DirectML devices."""
from __future__ import annotations

from typing import Any

import logging


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

        device = torch_directml.device()
        logger.info("Selected DirectML device: %s", device)
        return device
    except Exception as exc:  # pragma: no cover - best-effort fallback
        logger.warning("DirectML unavailable, falling back to CPU: %s", exc)
        try:
            import torch  # type: ignore

            return torch.device("cpu")
        except Exception:  # pragma: no cover - PyTorch missing
            return "cpu"
