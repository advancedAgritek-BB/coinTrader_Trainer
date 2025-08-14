from __future__ import annotations
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
