import os
import sys
import types
import asyncio

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the module under test which expects a package name
import data_loader
import feature_engineering

module = types.ModuleType("coinTrader_Trainer")
module.data_loader = data_loader
module.feature_engineering = feature_engineering
sys.modules["coinTrader_Trainer"] = module
sys.modules["coinTrader_Trainer.data_loader"] = data_loader
sys.modules["coinTrader_Trainer.feature_engineering"] = feature_engineering

import federated_trainer as ft


class FakeBooster:
    def predict(self, X, num_iteration=None):
        return np.zeros((len(X), 3))


def test_federated_gpu_training(monkeypatch, tmp_path):
    df = pd.DataFrame({"ts": range(6), "target": [0, 1] * 3, "f": range(6)})
    captured = {}

    async def fake_fetch(table, start, end):
        captured["fetch"] = True
        return df

    def fake_features(data, *args, **kwargs):
        captured["use_gpu"] = kwargs.get("use_gpu")
        return data

    uploads = []

    class DummyBucket:
        def upload(self, path, file_obj):
            uploads.append(path)

    class DummyClient:
        def __init__(self):
            self.storage = types.SimpleNamespace(from_=lambda b: DummyBucket())

    def fake_create(url, key):
        captured["create_client"] = (url, key)
        return DummyClient()

    def fake_train(params, dataset, *a, **k):
        captured["device_type"] = params.get("device_type")
        return FakeBooster()

    monkeypatch.setattr(ft, "fetch_data_range_async", fake_fetch)
    monkeypatch.setattr(ft, "make_features", fake_features)
    monkeypatch.setattr(ft.lgb, "train", fake_train)
    monkeypatch.setattr(ft, "create_client", fake_create)
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_KEY", "anon")

    asyncio.run(ft.train_federated_regime(None, None, num_clients=1))

    assert captured.get("device_type") == "gpu"
    assert captured.get("use_gpu") is True
    assert uploads == ["federated_model.pkl"]


def test_balanced_split(monkeypatch):
    df = pd.DataFrame({
        "ts": range(10),
        "target": [-1] + [0] * 2 + [1] * 7,
        "f": range(10),
    })
    captured_y = []

    async def fake_fetch(table, start, end):
        return df

    def fake_features(data, *a, **k):
        return data

    def fake_train_client(X, y, params):
        captured_y.append(y.reset_index(drop=True))
        return FakeBooster()

    monkeypatch.setattr(ft, "fetch_data_range_async", fake_fetch)
    monkeypatch.setattr(ft, "make_features", fake_features)
    monkeypatch.setattr(ft, "_train_client", fake_train_client)

    ft.train_federated_regime(None, None, num_clients=3)

    counts = [s.value_counts() for s in captured_y]
    for label in [-1, 0, 1]:
        label_counts = [c.get(label, 0) for c in counts]
        assert max(label_counts) - min(label_counts) <= 3

