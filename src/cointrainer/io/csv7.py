from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["ts", "open", "high", "low", "close", "volume", "trades"]

def read_csv7(path: Path | str) -> pd.DataFrame:
    """Read a CSV7 file (ts, o, h, l, c, v, trades) into a DataFrame."""
    df = pd.read_csv(path, names=COLUMNS, header=None)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    df = df.set_index("ts")
    return df
