import os
import sys
import types
import logging
import pytest
import swarm_sim
import data_loader

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ml_trainer


def test_cli_swarm_merges_params(monkeypatch):
    captured = {}

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        captured["params"] = params.copy()

        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(
        ml_trainer, "load_cfg", lambda path: {"regime_lgbm": {"objective": "binary"}}
    )
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (
            pd.DataFrame(np.random.normal(size=(4, 2))),
            pd.Series([0, 1, 0, 1]),
        ),
    )

    async def fake_run(start, end, *, table="ohlc_data"):
        captured["swarm_called"] = True
        captured["table"] = table
        return {"learning_rate": 0.1}

    module = types.SimpleNamespace(run_swarm_search=fake_run)
    monkeypatch.setitem(sys.modules, "swarm_sim", module)

    argv = ["ml_trainer", "train", "regime", "--swarm"]
    monkeypatch.setattr(sys, "argv", argv)

    ml_trainer.main()

    assert captured.get("swarm_called")
    assert captured["params"].get("learning_rate") == 0.1


@pytest.mark.asyncio
async def test_fetch_and_prepare_data_empty(monkeypatch, caplog):
    async def fake_fetch(table, start, end):
        return pd.DataFrame()

    monkeypatch.setattr(data_loader, "fetch_data_range_async", fake_fetch)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="No data available"):
            await swarm_sim.fetch_and_prepare_data("2021-01-01", "2021-01-02")

    assert "No data returned" in caplog.text
