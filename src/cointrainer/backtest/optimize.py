from __future__ import annotations

from pathlib import Path


def _train_one(
    csv_path: Path,
    symbol: str,
    horizon: int,
    hold: float,
    outdir: Path,
    device_type: str,
    max_bin: int,
    n_jobs: int | None,
    limit_rows: int | None,
) -> Path:
    """Train a single regime model and return its path.

    The real project performs feature engineering and model fitting here.
    For the purposes of the exercises we simply create a placeholder file so
    other components can operate without heavy dependencies.
    """

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / f"{symbol.lower()}_regime_lgbm.pkl"
    if not model_path.exists():
        model_path.write_bytes(b"placeholder")
    return model_path


__all__ = ["_train_one"]
