from __future__ import annotations

from pathlib import Path
from typing import Any


def backtest_csv(
    csv_path: Path,
    symbol: str,
    *,
    model_local: Path,
    outdir: Path,
    open_thr: float,
    fee_bps: float,
    slip_bps: float,
    position_mode: str = "gated",
) -> dict[str, Any]:
    """Minimal backtest stub.

    Reads ``csv_path`` and returns placeholder statistics. The real project
    performs a LightGBM model inference and trading simulation but that logic
    is intentionally trimmed for the exercises in these kata-style tests.
    """

    outdir.mkdir(parents=True, exist_ok=True)
    # A real implementation would load the model and run a proper backtest
    # against the CSV data.  Here we simply return a tiny statistics dict so
    # the CLI integration can function in tests without heavy dependencies.
    stats = {
        "trades": 0,
        "final_balance": 0.0,
    }
    return {"stats": stats}


__all__ = ["backtest_csv"]
