"""Helpers for batch training and CSV format detection."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def derive_symbol_from_filename(name: str) -> str:
    """Derive trading symbol from a filename.

    Examples
    --------
    >>> derive_symbol_from_filename("XRPUSD_1.csv")
    'XRPUSD'
    >>> derive_symbol_from_filename("ethusdt-1m.csv")
    'ETHUSDT'
    >>> derive_symbol_from_filename("ADAUSD.csv")
    'ADAUSD'
    """
    stem = Path(name).name
    stem = stem.split(".")[0]
    for delim in ("_", "-"):
        stem = stem.split(delim)[0]
    return stem.upper()


def is_csv7(path: str | Path) -> bool:
    """Return True if file appears to be a 7-column CSV without a header."""
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            first = f.readline().strip()
            if not first:
                return False
            parts = first.split(",")
            if len(parts) != 7:
                return False
            float(parts[0])  # ensure numeric timestamp
        return True
    except Exception:
        return False


def is_normalized_csv(path: str | Path) -> bool:
    """Return True if file has an OHLCV(+trades) header."""
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            header = f.readline().strip().lower().split(",")
        required: Iterable[str] = ["ts", "open", "high", "low", "close", "volume"]
        return all(col in header for col in required)
    except Exception:
        return False
