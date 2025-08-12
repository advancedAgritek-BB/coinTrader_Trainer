import os
import sys
from datetime import datetime

import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))

from cointrainer.data import loader


def test_loader_reads_parquet_cache(tmp_path, monkeypatch):
    df = pd.DataFrame({"ts": [1], "price": [2.0]})
    cache_file = tmp_path / "cache.parquet"
    df.to_parquet(cache_file)

    def fail(*a, **k):
        raise AssertionError("should not hit supabase")

    monkeypatch.setattr(loader, "select_range", fail)

    out = loader.fetch_trade_logs(
        datetime(2021, 1, 1), datetime(2021, 1, 2), "BTC", cache_path=str(cache_file)
    )
    pd.testing.assert_frame_equal(out, df)
