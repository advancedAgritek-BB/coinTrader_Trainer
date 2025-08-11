import logging
import os
import sys

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cointrainer.train import pipeline as train_pipeline


class DummyModel:
    def predict(self, X):
        return np.zeros((len(X), 3))


class DummyRegistry:
    def __init__(self, *a):
        pass

    def upload(self, model, name, metrics, conflict_key=None):
        pass


def test_gpu_helper_import_warning(monkeypatch, caplog):
    monkeypatch.delitem(sys.modules, "lightgbm_gpu_build", raising=False)

    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_KEY", "anon")

    monkeypatch.setattr(train_pipeline, "check_clinfo_gpu", lambda: True)

    monkeypatch.setattr(
        train_pipeline,
        "fetch_trade_logs",
        lambda s, e, **k: pd.DataFrame({"ts": [0], "target": [0]}),
    )
    monkeypatch.setattr(train_pipeline, "make_features", lambda d: d)
    captured = {}

    def fake_train(X, y, p, use_gpu=True):
        captured["gpu"] = use_gpu
        return DummyModel(), {}

    monkeypatch.setattr(train_pipeline, "train_regime_lgbm", fake_train)
    monkeypatch.setattr(train_pipeline, "simulate_signal_pnl", lambda df, preds, **k: {})
    monkeypatch.setattr(train_pipeline, "ModelRegistry", DummyRegistry)

    monkeypatch.setattr(sys, "argv", ["prog"])

    with caplog.at_level(logging.WARNING):
        train_pipeline.main()

    assert any("GPU wheel helper unavailable" in r.message for r in caplog.records)


def test_main_falls_back_to_cpu(monkeypatch, caplog):
    monkeypatch.delitem(sys.modules, "lightgbm_gpu_build", raising=False)

    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_KEY", "anon")

    monkeypatch.setattr(train_pipeline, "check_clinfo_gpu", lambda: False)
    monkeypatch.setattr(
        train_pipeline,
        "fetch_trade_logs",
        lambda s, e, **k: pd.DataFrame({"ts": [0], "target": [0]}),
    )
    monkeypatch.setattr(train_pipeline, "make_features", lambda d: d)
    captured = {}

    def fake_train2(X, y, p, use_gpu=True):
        captured["gpu"] = use_gpu
        return DummyModel(), {}

    monkeypatch.setattr(train_pipeline, "train_regime_lgbm", fake_train2)
    monkeypatch.setattr(train_pipeline, "simulate_signal_pnl", lambda df, preds, **k: {})
    monkeypatch.setattr(train_pipeline, "ModelRegistry", DummyRegistry)

    monkeypatch.setattr(sys, "argv", ["prog"])

    with caplog.at_level(logging.WARNING):
        train_pipeline.main()
    assert any("falling back to CPU" in r.message for r in caplog.records)
    assert captured.get("gpu") is False
