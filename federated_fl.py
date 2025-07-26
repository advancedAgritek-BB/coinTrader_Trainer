"""Placeholder true federated learning helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

__all__ = ["start_server", "start_client"]


def start_server(
    start_ts: str,
    end_ts: str,
    *,
    config_path: str = "cfg.yaml",
    params_override: Optional[Dict[str, Any]] = None,
    table: str = "ohlc_data",
) -> Tuple[None, Dict[str, Any]]:
    """Start a federated learning server.

    This is a lightweight stub used during testing. Real implementations
    should launch the FL server and return any training metrics.
    """
    return None, {}


def start_client(
    start_ts: str,
    end_ts: str,
    *,
    config_path: str = "cfg.yaml",
    params_override: Optional[Dict[str, Any]] = None,
    table: str = "ohlc_data",
) -> Tuple[None, Dict[str, Any]]:
    """Start a federated learning client.

    This stub simply returns empty metrics and is intended to be mocked in
    tests.
    """
    return None, {}
