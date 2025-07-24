import os
import sys
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import train_pipeline


class DummyModel:
    def predict(self, X):
        return pd.Series([0 for _ in range(len(X))])


class DummyRegistry:
    def __init__(self, *a):
        pass

    def upload(self, model, name, metrics):
        pass


def test_gpu_helper_import_warning(monkeypatch, caplog):
    monkeypatch.delitem(sys.modules, "lightgbm_gpu_build", raising=False)

    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_KEY", "anon")

    monkeypatch.setattr(train_pipeline, "fetch_trade_logs", lambda s, e: pd.DataFrame({"ts": [0], "target": [0]}))
    monkeypatch.setattr(train_pipeline, "make_features", lambda d: d)
    monkeypatch.setattr(train_pipeline, "train_regime_lgbm", lambda X, y, p, use_gpu=True: (DummyModel(), {}))
    monkeypatch.setattr(train_pipeline, "simulate_signal_pnl", lambda df, preds: 0.0)
    monkeypatch.setattr(train_pipeline, "ModelRegistry", DummyRegistry)

    monkeypatch.setattr(sys, "argv", ["prog"])

    with caplog.at_level(logging.WARNING):
        train_pipeline.main()

    assert any("GPU wheel helper unavailable" in r.message for r in caplog.records)
