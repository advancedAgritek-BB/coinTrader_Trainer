from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {"open", "high", "low", "close"}


def load_market_csv(path: str) -> pd.DataFrame:
    """Return OHLC market data from ``path``.

    The CSV file must contain at least ``open``, ``high``, ``low`` and
    ``close`` columns and a timestamp column. The timestamp column is parsed
    to a ``DatetimeIndex`` (UTC) and the resulting frame is sorted.
    """
    df = pd.read_csv(path)

    # Determine timestamp column
    ts_col = None
    for candidate in ("timestamp", "ts", "date", "time"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = df.columns[0]

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.set_index(ts_col).sort_index()

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    return df[["open", "high", "low", "close"]]
