import os
import sys
import logging
import pytest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import types
import data_loader
import feature_engineering

module = types.SimpleNamespace(data_loader=data_loader, feature_engineering=feature_engineering)
sys.modules["coinTrader_Trainer"] = module
sys.modules["coinTrader_Trainer.data_loader"] = data_loader
sys.modules["coinTrader_Trainer.feature_engineering"] = feature_engineering

import federated_trainer as ft

from trainers import federated
from trainers.federated import train_federated_regime


class FakeBooster:
    def predict(self, X, num_iteration=None):
        return np.zeros(len(X))


def test_train_federated_regime_returns_callable_and_uploads(monkeypatch):
    df = pd.DataFrame({"f": [1, 2, 3], "target": [-1, 0, 1]})
    calls = {}

    async def fake_fetch(table, start, end):
        calls["fetch"] = (table, start, end)
        return df

    def fake_features(data, *a, **k):
        calls["features"] = True
        return data

    def fake_train(params, dataset, *a, **k):
        calls["train"] = True
        return FakeBooster()

    class DummyRegistry:
        def __init__(self, url, key):
            calls["registry_init"] = (url, key)

        def upload(self, model, name, metrics):
            calls["uploaded"] = metrics

    monkeypatch.setattr(federated, "fetch_data_range_async", fake_fetch)
    monkeypatch.setattr(federated, "make_features", fake_features)
    monkeypatch.setattr(federated.lgb, "train", fake_train)
    monkeypatch.setattr(federated, "ModelRegistry", DummyRegistry)
    monkeypatch.setenv("SUPABASE_URL", "http://localhost")
    monkeypatch.setenv("SUPABASE_KEY", "anon")

    start = "2021-01-01"
    end = "2021-01-02"
    ensemble, metrics = train_federated_regime(start, end, {"objective": "reg"})

    assert callable(ensemble)
    assert "uploaded" in calls
    assert metrics["n_models"] == 1
    assert calls["fetch"] == ("ohlc_data", start, end)


def test_train_federated_regime_empty_data_raises(monkeypatch, caplog):
    async def fake_fetch(table, start, end):
        return pd.DataFrame()

    monkeypatch.setattr(ft, "fetch_data_range_async", fake_fetch)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="No data available"):
            ft.train_federated_regime("2021-01-01", "2021-01-02")

    assert "No data returned" in caplog.text
