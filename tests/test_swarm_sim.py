"""Tests for the swarm simulation utilities and CLI integration."""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import lightgbm as lgb
import data_loader
import ml_trainer
from swarm_sim import run_swarm_simulation, Agent


class DummyBooster:
    best_iteration = 1

    def predict(self, data, num_iteration=None):
        return np.zeros(len(data))


async def _fake_fetch(*args, **kwargs):
    return pd.DataFrame({"price": [1, 2], "target": [0, 1]})


def _fake_train(params, dataset, num_boost_round=1):
    return DummyBooster()


@pytest.mark.asyncio
async def test_run_swarm_simulation_updates_agents(monkeypatch):
    monkeypatch.setattr(data_loader, "fetch_data_range_async", _fake_fetch)
    monkeypatch.setattr(lgb, "train", _fake_train)

    params, agents = await run_swarm_simulation(num_agents=2)

    assert isinstance(params, dict)
    assert len(agents) == 2
    assert all(isinstance(a, Agent) and a.fitness == 1.0 for a in agents)


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

    import swarm_sim

    monkeypatch.setattr(swarm_sim, "run_swarm_search", fake_swarm)

    monkeypatch.setattr(sys, "argv", ["prog", "train", "regime", "--swarm"])
    ml_trainer.main()

    assert captured["params"] == {"a": 1, "b": 2}

