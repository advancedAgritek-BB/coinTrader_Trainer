"""Tests for the swarm simulation utilities and CLI integration."""

import os
import sys
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ml_trainer
import swarm_sim
from swarm_sim import run_swarm_simulation
from swarm_sim import run_swarm_simulation, Agent
import swarm_sim
from datetime import datetime


class DummyBooster:
    best_iteration = 1

    def predict(self, data, num_iteration=None):
        return np.zeros(len(data))


async def _fake_fetch(*args, **kwargs):
    return pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=2, freq="1D"),
            "price": [1, 2],
            "high": [1, 2],
            "low": [0, 1],
            "target": [0, 1],
        }
    )


train_calls = []


def _fake_train(*args, **kwargs):
    train_calls.append(1)
    n = 30
    return pd.DataFrame({
        "ts": pd.date_range("2021-01-01", periods=n, freq="1T"),
        "price": np.arange(n, dtype=float),
        "high": np.arange(n, dtype=float) + 0.5,
        "low": np.arange(n, dtype=float) - 0.5,
        "target": np.random.randint(0, 2, size=n),
        "ts": pd.date_range("2021-01-01", periods=n, freq="h"),
        "price": np.linspace(1, n, n),
        "high": np.linspace(1, n, n),
        "low": np.linspace(0, n - 1, n),
        "target": [0, 1] * (n // 2) + [0] * (n % 2),
    })


def _fake_train(params, dataset, num_boost_round=1, **kwargs):
    return DummyBooster()


@pytest.mark.asyncio
async def test_run_swarm_simulation_updates_agents(monkeypatch):
    from datetime import datetime

    import importlib, sys
    module = importlib.import_module("swarm_sim")
    if not hasattr(module, "yaml"):
        sys.modules.pop("swarm_sim", None)
        module = importlib.import_module("swarm_sim")
    import lightgbm as lgb
    from swarm_sim import run_swarm_simulation
    monkeypatch.setattr(module, "fetch_data_range_async", _fake_fetch)
    monkeypatch.setattr(lgb, "train", _fake_train)
    monkeypatch.setattr(module, "make_features", lambda df: df)
    monkeypatch.setattr(module.yaml, "safe_load", lambda f: {})

    params = await run_swarm_simulation(
        datetime(2021, 1, 1), datetime(2021, 1, 2), num_agents=2
    monkeypatch.setattr(data_loader, "fetch_data_range_async", _fake_fetch)
    monkeypatch.setattr(swarm_sim, "fetch_data_range_async", _fake_fetch)
    monkeypatch.setattr(lgb, "train", _fake_train)
    monkeypatch.setattr(swarm_sim, "evolve_swarm", lambda *a, **k: None)

    params = await swarm_sim.run_swarm_simulation(datetime.utcnow(), datetime.utcnow(), num_agents=2)
    monkeypatch.setattr(swarm_sim, "evolve_swarm", lambda a, g: None)

    params = await run_swarm_simulation(
        datetime(2020, 1, 1), datetime(2020, 1, 2), num_agents=2
    )

        datetime(2021, 1, 1), datetime(2021, 1, 2), num_agents=2
    )
    assert isinstance(params, dict)
    assert train_calls


def test_ml_trainer_swarm_merges(monkeypatch):
    captured = {}

    def fake_trainer(X, y, params, use_gpu=False):
        captured["params"] = params
        return None, {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_trainer, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "_make_dummy_data", lambda: (pd.DataFrame([[1]]), pd.Series([0])))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {"a": 1}})

    async def fake_swarm(start, end):
        return {"b": 2}
    async def fake_swarm(*args, **kwargs):
        return {"b": 2}
    async def fake_swarm(start, end):
        return {"b": 2}, [Agent({})]

    import swarm_sim

    monkeypatch.setattr(swarm_sim, "run_swarm_search", fake_swarm)

    monkeypatch.setattr(sys, "argv", ["prog", "train", "regime", "--swarm"])
    ml_trainer.main()

    assert captured["params"] == {"a": 1, "b": 2}







