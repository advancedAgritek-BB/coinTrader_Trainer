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
from __future__ import annotations

from pathlib import Path


def iter_csv_files(folder: str | Path, glob: str = "*.csv", recursive: bool = False) -> list[Path]:
    """Return a list of CSV files under *folder* matching *glob*.

    Parameters
    ----------
    folder: str | Path
        Root directory to search.
    glob: str
        Glob pattern to match files. Defaults to ``"*.csv"``.
    recursive: bool
        If ``True`` search subdirectories recursively.
    """
    root = Path(folder)
    if not root.exists():
        return []
    files = list(root.rglob(glob)) if recursive else list(root.glob(glob))
    return [f for f in files if f.is_file()]


def is_csv7(path: str | Path) -> bool:
    """Return ``True`` if the file appears to be a headerless 7-column CSV."""
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            first = fh.readline().strip()
    except Exception:
        return False
    parts = first.split(",")
    if len(parts) != 7:
        return False
    for p in parts:
        try:
            float(p)
        except ValueError:
            return False
    return True


def is_normalized_csv(path: str | Path) -> bool:
    """Return ``True`` if the file looks like a normalized OHLCV CSV."""
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            header = fh.readline().lower()
    except Exception:
        return False
    return all(h in header for h in ["open", "high", "low", "close", "volume"])


def derive_symbol(path: Path, mode: str = "filename", fixed: str | None = None) -> str:
    """Derive a trading symbol from *path* according to *mode*.

    ``mode`` may be ``"filename"`` (default) which uses the stem of the
    filename, ``"parent"`` which uses the name of the parent directory, or
    ``"fixed"`` which returns ``fixed``.
    """
    if mode == "filename":
        return path.stem.split("_")[0].upper()
    if mode == "parent":
        return path.parent.name.upper()
    if mode == "fixed":
        if not fixed:
            raise ValueError("symbol must be provided when mode='fixed'")
        return fixed.upper()
    raise ValueError(f"Unknown derive mode: {mode}")
from pathlib import Path
from typing import List, Literal
import re
import pandas as pd


def iter_csv_files(folder: str | Path, glob: str = "*.csv", recursive: bool = False) -> List[Path]:
    root = Path(folder)
    it = root.rglob(glob) if recursive else root.glob(glob)
    files = [p for p in it if p.is_file()]
    files.sort()
    return files


def is_csv7(path: str | Path, probe_rows: int = 3) -> bool:
    """
    Heuristically detect a headerless 7-col CSV:
      ts, open, high, low, close, volume, trades
    """
    p = Path(path)
    try:
        df = pd.read_csv(p, header=None, nrows=probe_rows)
        return df.shape[1] == 7
    except Exception:
        return False


def is_normalized_csv(path: str | Path, probe_rows: int = 3) -> bool:
    """
    Detect a normalized OHLCV(+trades) CSV with a header and a datetime index in col 0.
    """
    p = Path(path)
    try:
        df = pd.read_csv(p, nrows=probe_rows)
        cols = [c.lower() for c in df.columns]
        needed = {"open","high","low","close","volume"}
        return needed.issubset(set(cols))
    except Exception:
        return False


def derive_symbol_from_filename(path: str | Path) -> str:
    """
    Derive a symbol like 'XRPUSD' from filenames such as:
      XRPUSD_1.csv, ethusdt-1m.csv, ADAUSD.csv
    Rule: take the stem, strip trailing timeframe tokens and non-alnum, uppercase.
    """
    stem = Path(path).stem
    # strip common timeframe suffixes like _1m, -1m, _1, -1, etc.
    stem = re.sub(r"([_\-\.]?\d+[a-zA-Z]*)+$", "", stem)
    # keep alnum only
    stem = re.sub(r"[^A-Za-z0-9]", "", stem)
    return stem.upper() or "UNKN"


def derive_symbol(path: str | Path, mode: Literal["filename","parent","fixed"]="filename", fixed: str | None = None) -> str:
    p = Path(path)
    if mode == "fixed" and fixed:
        return fixed.upper()
    if mode == "parent":
        return re.sub(r"[^A-Za-z0-9]", "", p.parent.name).upper() or "UNKN"
    return derive_symbol_from_filename(p)
