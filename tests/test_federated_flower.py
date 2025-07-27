import os
import sys
import types
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

sys.modules.setdefault(
    "supabase",
    types.SimpleNamespace(
        create_client=lambda *a, **k: object(),
        Client=object,
        SupabaseException=Exception,
    ),
)

# Provide dummy package for relative imports
data_loader = types.ModuleType("data_loader")
data_loader.fetch_data_range_async = lambda *a, **k: None
data_loader.fetch_trade_aggregates = lambda *a, **k: None
data_loader._get_redis_client = lambda: None
feature_engineering = types.ModuleType("feature_engineering")
feature_engineering.make_features = lambda *a, **k: None
pkg = types.ModuleType("coinTrader_Trainer")
pkg.data_loader = data_loader
pkg.feature_engineering = feature_engineering
sys.modules["coinTrader_Trainer"] = pkg
sys.modules["coinTrader_Trainer.data_loader"] = data_loader
sys.modules["coinTrader_Trainer.feature_engineering"] = feature_engineering

import federated_fl as flmod


class DummyFlower:
    def __init__(self):
        self.client = types.SimpleNamespace(start_numpy_client=lambda *a, **k: None)
        self.server = types.SimpleNamespace(start_server=lambda *a, **k: None)
        self.simulation = types.SimpleNamespace(start_simulation=lambda *a, **k: None)




def test_start_client_invokes_flower(monkeypatch):
    called = {}

    def fake_load(path):
        return {"objective": "binary"}

    def fake_prepare(start, end, *, table="ohlc_data"):
        return pd.DataFrame({"f": [1, 2]}), pd.Series([0, 1])

    def fake_start(address, client):
        called["address"] = address
        called["client"] = client

    monkeypatch.setattr(flmod, "_HAVE_FLWR", True, raising=False)
    dummy = DummyFlower()
    monkeypatch.setattr(flmod, "fl", dummy, raising=False)
    monkeypatch.setattr(flmod, "_load_params", fake_load)
    monkeypatch.setattr(flmod, "_prepare_data", fake_prepare)
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

    monkeypatch.setattr(flmod, "_HAVE_FLWR", True, raising=False)
    dummy = DummyFlower()
    monkeypatch.setattr(flmod, "fl", dummy, raising=False)
    monkeypatch.setattr(flmod.fl.server, "start_server", fake_start)
    monkeypatch.setattr(flmod, "_load_params", fake_load)
    monkeypatch.setattr(flmod, "_prepare_data", fake_prepare)
    monkeypatch.setattr(flmod.lgb, "Booster", DummyBooster)
    monkeypatch.setattr(flmod.joblib, "dump", lambda *a, **k: called.setdefault("dump", True))

    model, metrics = flmod.start_server("s", "e", num_rounds=1)

    assert called["address"] == "0.0.0.0:8080"
    assert called["config"].num_rounds == 1
    assert isinstance(model, flmod.FederatedEnsemble)
    assert metrics["n_models"] == 1
