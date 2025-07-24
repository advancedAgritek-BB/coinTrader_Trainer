"""Tests for the swarm simulation utilities and CLI integration."""

import os
import sys
import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ml_trainer
import swarm_sim


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


def _fake_train(params, dataset, num_boost_round=1, **kwargs):
    train_calls.append(params)
    return DummyBooster()


@pytest.mark.asyncio
async def test_run_swarm_search_updates_params(monkeypatch):
    monkeypatch.setattr(swarm_sim, "fetch_data_range_async", _fake_fetch)
    import data_loader
    monkeypatch.setattr(data_loader, "fetch_data_range_async", _fake_fetch)
    monkeypatch.setattr(swarm_sim, "make_features", lambda df: df)
    monkeypatch.setattr(swarm_sim.lgb, "train", _fake_train)
    monkeypatch.setattr(swarm_sim.yaml, "safe_load", lambda f: {})

    params = await swarm_sim.run_swarm_search(
        datetime(2021, 1, 1), datetime(2021, 1, 2), num_agents=2
    )

    assert train_calls
    assert isinstance(params, dict)


async def fake_swarm(*args, **kwargs):
    return {"b": 2}


def test_ml_trainer_swarm_merges(monkeypatch):
    captured = {}

    def fake_trainer(X, y, params, use_gpu=False):
        captured["params"] = params
        return None, {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_trainer, "regime_lgbm"))
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (pd.DataFrame([[1]]), pd.Series([0])),
    )
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {"a": 1}})
    monkeypatch.setitem(sys.modules, "swarm_sim", swarm_sim)
    monkeypatch.setattr(swarm_sim, "run_swarm_search", fake_swarm)
    monkeypatch.setattr(swarm_sim, "run_swarm_simulation", fake_swarm, raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "train", "regime", "--swarm"])

    ml_trainer.main()

    assert captured["params"] == {"a": 1, "b": 2}
