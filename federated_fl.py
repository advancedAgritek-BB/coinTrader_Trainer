"""Flower-based federated training utilities (stub)."""
from __future__ import annotations

from typing import Optional


def launch(start_ts: str, end_ts: str, *, config_path: str = "cfg.yaml", table: str = "ohlc_data", params_override: Optional[dict] = None) -> None:
    """Placeholder launcher for Flower-based federated training."""
    # In the real implementation this would start the Flower server and clients
    # but for testing we simply return without doing anything.
    return None
