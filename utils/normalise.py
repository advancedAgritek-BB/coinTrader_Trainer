from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

_RENAME_MAP: Mapping[str, str] = {
    "timestamp": "ts",
    "date": "ts",
    "unix": "ts",
    "time": "ts",
    "ts": "ts",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "price",
    "price": "price",
    "vwap": "vwap",
    "volume": "volume",
    "volume usdt": "volume_usdt",
    "volume xrp": "volume_xrp",
    "tradecount": "tradecount",
    "trades": "trades",
    "symbol": "symbol",
    "exchange": "exchange",
}


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with standardised OHLC column names."""
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        key = col.lower()
        if key in _RENAME_MAP:
            new_cols[col] = _RENAME_MAP[key]
    df = df.rename(columns=new_cols)

    # drop duplicate columns
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    if "price" not in df.columns and "close" in df.columns:
        df = df.rename(columns={"close": "price"})
    if "price" in df.columns and "close" not in df.columns:
        df["close"] = df["price"]

    return df
