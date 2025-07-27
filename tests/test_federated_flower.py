import os
import sys
import types
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

sys.modules.setdefault(
    "supabase",
    types.SimpleNamespace(create_client=lambda *a, **k: object(), Client=object),
)

# Provide dummy package for relative imports
import data_loader
import feature_engineering
pkg = types.ModuleType("coinTrader_Trainer")
pkg.data_loader = data_loader
pkg.feature_engineering = feature_engineering
sys.modules["coinTrader_Trainer"] = pkg
sys.modules["coinTrader_Trainer.data_loader"] = data_loader
sys.modules["coinTrader_Trainer.feature_engineering"] = feature_engineering

import federated_fl as flmod


def test_start_client_invokes_flower(monkeypatch):
    called = {}

    def fake_load(path):
        return {"objective": "binary"}

    def fake_prepare(start, end, *, table="ohlc_data"):
        return pd.DataFrame({"f": [1, 2]}), pd.Series([0, 1])

    def fake_start(address, client):
        called["address"] = address
        called["client"] = client

    monkeypatch.setattr(flmod, "_load_params", fake_load)
    monkeypatch.setattr(flmod, "prepare_data", fake_prepare)
    monkeypatch.setattr(flmod.fl.client, "start_numpy_client", fake_start)

    flmod.start_client("s", "e", server_address="server:1234", params_override={"lr": 0.1})

    assert called["address"] == "server:1234"
    assert isinstance(called["client"], flmod._LGBClient)


def test_start_server_invokes_flower(monkeypatch):
    called = {}

    def fake_start(address, *, config, strategy):
        called["address"] = address
        called["config"] = config
        called["strategy"] = strategy
        strategy.models = [b"model"]

    class DummyBooster:
        def __init__(self, model_str=None):
            pass

        def predict(self, X):
            return np.zeros((len(X), 2))

    def fake_load(path):
        return {}

    def fake_prepare(start, end, *, table="ohlc_data"):
        return pd.DataFrame({"f": [0, 1]}), pd.Series([0, 1])

    monkeypatch.setattr(flmod.fl.server, "start_server", fake_start)
    monkeypatch.setattr(flmod, "_load_params", fake_load)
    monkeypatch.setattr(flmod, "prepare_data", fake_prepare)
    monkeypatch.setattr(flmod.lgb, "Booster", DummyBooster)
    monkeypatch.setattr(flmod.joblib, "dump", lambda *a, **k: called.setdefault("dump", True))

    model, metrics = flmod.start_server("s", "e", num_rounds=1)

    assert called["address"] == "0.0.0.0:8080"
    assert called["config"].num_rounds == 1
    assert isinstance(model, flmod.FederatedEnsemble)
    assert metrics["n_models"] == 1
