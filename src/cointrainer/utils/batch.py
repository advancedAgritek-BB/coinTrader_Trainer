"""Utility helpers for batch CSV training and symbol derivation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd


def iter_csv_files(folder: str | Path, glob: str = "*.csv", recursive: bool = False) -> list[Path]:
    """Return a list of CSV files under *folder* matching *glob*."""

    root = Path(folder)
    it = root.rglob(glob) if recursive else root.glob(glob)
    return [p for p in it if p.is_file()]


def is_csv7(path: str | Path, probe_rows: int = 3) -> bool:
    """Heuristically detect a headerless 7-column CSV."""

    p = Path(path)
    try:
        df = pd.read_csv(p, header=None, nrows=probe_rows)
        return df.shape[1] == 7
    except Exception:
        return False


def is_normalized_csv(path: str | Path, probe_rows: int = 3) -> bool:
    """Return True if the file looks like a normalized OHLCV(+trades) CSV."""

    p = Path(path)
    try:
        df = pd.read_csv(p, nrows=probe_rows)
        cols = [c.lower() for c in df.columns]
        needed = {"open", "high", "low", "close", "volume"}
        return needed.issubset(set(cols))
    except Exception:
        return False


def derive_symbol_from_filename(path: str | Path) -> str:
    """Derive a symbol like 'XRPUSD' from typical CSV filenames."""

    stem = Path(path).stem
    stem = re.sub(r"([_\-\.]?\d+[a-zA-Z]*)+$", "", stem)
    stem = re.sub(r"[^A-Za-z0-9]", "", stem)
    return stem.upper() or "UNKN"


def derive_symbol(
    path: Path, mode: Literal["filename", "parent", "fixed"] = "filename", fixed: str | None = None
) -> str:
    """Derive a trading symbol from *path* according to *mode*."""

    if mode == "fixed" and fixed:
        return fixed.upper()
    if mode == "parent":
        return re.sub(r"[^A-Za-z0-9]", "", path.parent.name).upper() or "UNKN"
    return derive_symbol_from_filename(path)

