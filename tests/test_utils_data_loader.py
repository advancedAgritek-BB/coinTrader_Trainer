from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import load_market_csv


def test_load_market_csv(tmp_path):
    csv_file = tmp_path / "market.csv"
    csv_file.write_text(
        "timestamp,open,high,low,close\n2021-01-01T00:00:00Z,1,2,0.5,1.5\n"
    )

    df = load_market_csv(str(csv_file))

    assert list(df.columns) == ["open", "high", "low", "close"]
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert float(df.iloc[0]["close"]) == 1.5
