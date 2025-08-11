import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cointrainer.train import pipeline as train_pipeline


def test_default_window_used(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("default_window_days: 3\n")

    class Args:
        pass

    Args.cfg = str(cfg_path)
    Args.start_ts = None
    Args.end_ts = None
    Args.table = "ohlc_data"

    monkeypatch.setattr(train_pipeline, "parse_args", lambda: Args)
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_KEY", "anon")
    monkeypatch.setattr(train_pipeline, "check_clinfo_gpu", lambda: True)

    fixed_now = datetime(2021, 1, 10)

    class DummyDatetime:
        @classmethod
        def utcnow(cls):
            return fixed_now

    monkeypatch.setattr(train_pipeline, "datetime", DummyDatetime)

    captured = {}

    def fake_fetch(start, end, **k):
        captured["start"] = start
        captured["end"] = end
        return pd.DataFrame({"ts": [start], "target": [0]})

    monkeypatch.setattr(train_pipeline, "fetch_trade_logs", fake_fetch)
    monkeypatch.setattr(train_pipeline, "make_features", lambda df: df)

    class FakeModel:
        def predict(self, data):
            return np.zeros((len(data), 3))

    monkeypatch.setattr(
        train_pipeline,
        "train_regime_lgbm",
        lambda X, y, p, use_gpu=True: (FakeModel(), {}),
    )
    monkeypatch.setattr(train_pipeline, "simulate_signal_pnl", lambda df, signal, **k: {})

    class DummyRegistry:
        def __init__(self, url, key):
            pass

        def upload(self, model, name, metrics, conflict_key=None):
            captured["uploaded"] = True

    monkeypatch.setattr(train_pipeline, "ModelRegistry", DummyRegistry)

    train_pipeline.main()

    assert captured["start"] == fixed_now - timedelta(days=3)
