import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
