import os
import sys
import types
import asyncio
import pandas as pd
import numpy as np
import httpx
import pytest
import sklearn.model_selection

# Ensure package modules resolve like other tests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data_loader
import feature_engineering

module = types.ModuleType("coinTrader_Trainer")
module.data_loader = data_loader
module.feature_engineering = feature_engineering
sys.modules["coinTrader_Trainer"] = module
sys.modules["coinTrader_Trainer.data_loader"] = data_loader
sys.modules["coinTrader_Trainer.feature_engineering"] = feature_engineering

import federated_trainer as ft
from trainers import federated as fed
from trainers import regime_lgbm
import federated_fl
import swarm_sim


class FakeBooster:
    def __init__(self, *a, **k):
        pass

    best_iteration = 1

    def predict(self, X, num_iteration=None):
        return np.zeros((len(X), 3))


def _simple_data():
    X = pd.DataFrame({"f": range(15)})
    y = pd.Series([0, 1, 2] * 5)
    return X, y


def test_federated_trainer_upload_failure(monkeypatch):
    X, y = _simple_data()

    async def fake_prepare(*a, **k):
        return X, y

    monkeypatch.setattr(ft, "fetch_trade_aggregates", lambda *a, **k: None)
    monkeypatch.setattr(ft, "_prepare_data", fake_prepare)
    monkeypatch.setattr(ft, "_train_client", lambda *a, **k: FakeBooster())

    class Bucket:
        def upload(self, path, fh):
            raise httpx.HTTPError("boom")

    class Client:
        def __init__(self):
            self.storage = types.SimpleNamespace(from_=lambda b: Bucket())

    monkeypatch.setattr(ft, "create_client", lambda u, k: Client())
    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_KEY", "key")

    with pytest.raises(httpx.HTTPError):
        asyncio.run(ft.train_federated_regime("2021-01-01", "2021-01-02", num_clients=1))


def test_federated_module_env_upload_failure(monkeypatch):
    X, y = _simple_data()

    async def fake_fetch(table, start, end):
        return pd.concat([X, y.rename("target")], axis=1)

    monkeypatch.setattr(fed, "fetch_data_range_async", fake_fetch)
    monkeypatch.setattr(fed, "make_features", lambda d, **k: d)
    monkeypatch.setattr(fed.lgb, "train", lambda *a, **k: FakeBooster())

    class DummyRegistry:
        def __init__(self, url, key):
            pass

        def upload(self, *a, **k):
            raise httpx.HTTPError("fail")

    monkeypatch.setattr(fed, "ModelRegistry", DummyRegistry)
    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_KEY", "key")

    with pytest.raises(httpx.HTTPError):
        fed.train_federated_regime("s", "e", {"objective": "reg"}, n_clients=1)


def test_regime_lgbm_upload_failure(monkeypatch):
    X, y = _simple_data()

    monkeypatch.setattr(regime_lgbm.lgb, "train", lambda *a, **k: FakeBooster())
    monkeypatch.setattr(regime_lgbm, "StratifiedKFold", lambda *a, **k: sklearn.model_selection.StratifiedKFold(n_splits=2))

    class DummyRegistry:
        def __init__(self, url, key):
            pass

        def upload(self, *a, **k):
            raise httpx.HTTPError("fail")

    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_KEY", "key")
    monkeypatch.setattr(regime_lgbm, "ModelRegistry", DummyRegistry)

    with pytest.raises(httpx.HTTPError):
        regime_lgbm.train_regime_lgbm(X, y, {"objective": "reg"}, use_gpu=False)


def test_federated_fl_upload_failure(monkeypatch):
    X, y = _simple_data()

    monkeypatch.setattr(federated_fl, "_HAVE_FLWR", True, raising=False)

    def fake_start_simulation(*a, **k):
        strategy = k.get("strategy") or a[0]
        strategy.models = [b"m"]

    dummy = types.SimpleNamespace(
        simulation=types.SimpleNamespace(start_simulation=fake_start_simulation)
    )
    monkeypatch.setattr(federated_fl, "fl", dummy, raising=False)
    monkeypatch.setattr(federated_fl, "_load_params", lambda p: {"objective": "reg"})
    monkeypatch.setattr(federated_fl, "prepare_data", lambda *a, **k: (X, y))
    monkeypatch.setattr(federated_fl.lgb, "Booster", FakeBooster)

    class Bucket:
        def upload(self, path, fh):
            raise httpx.HTTPError("fail")

    class Client:
        def __init__(self):
            self.storage = types.SimpleNamespace(from_=lambda b: Bucket())

    monkeypatch.setattr(federated_fl, "create_client", lambda u, k: Client())
    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_KEY", "key")

    with pytest.raises(httpx.HTTPError):
        federated_fl.launch("s", "e", num_clients=1, num_rounds=1)


@pytest.mark.asyncio
async def test_swarm_upload_failure(monkeypatch):
    X, y = _simple_data()

    async def fake_fetch(*a, **k):
        return X, y

    async def fake_sim(self, X, y, base):
        self.fitness = 1.0

    monkeypatch.setattr(swarm_sim, "fetch_and_prepare_data", fake_fetch)
    monkeypatch.setattr(swarm_sim.SwarmAgent, "simulate", fake_sim)
    monkeypatch.setattr(swarm_sim.yaml, "safe_load", lambda fh: {})

    class DummyReg:
        def __init__(self, *a, **k):
            pass

        def upload_dict(self, *a, **k):
            raise httpx.HTTPError("boom")

    monkeypatch.setattr(swarm_sim, "ModelRegistry", DummyReg)
    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_KEY", "key")

    with pytest.raises(httpx.HTTPError):
        await swarm_sim.run_swarm_search(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02"), num_agents=1)
