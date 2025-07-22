"""Utilities for working with DirectML devices."""
from __future__ import annotations

from typing import Any


def get_dml_device() -> Any:
    """Return a DirectML ``device`` or CPU fallback.

    The function attempts to create a ``torch_directml.device`` instance.
    If the package is not available or any error occurs, a CPU device is
    returned.  When PyTorch itself is not installed, the string ``"cpu"``
    is returned.
    """
    try:
        import torch_directml  # type: ignore

        return torch_directml.device()
    except Exception:
        try:
            import torch

            return torch.device("cpu")
        except Exception:
            return "cpu"
