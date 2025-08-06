from __future__ import annotations

import pandas as pd


def load_market_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
    df.set_index("timestamp", inplace=True)
    return df
