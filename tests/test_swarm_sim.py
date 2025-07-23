import os
import sys
import types
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ml_trainer

class DummyBooster:
    best_iteration = 1
    def predict(self, data, num_iteration=None):
        return np.zeros(len(data))

async def fake_swarm_search(start, end):
    return {"lr": 0.1}

def test_ml_trainer_swarm_merges(monkeypatch):
    captured = {}

    def fake_trainer(X, y, params, use_gpu=False):
        captured["params"] = params
        return DummyBooster(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_trainer, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "_make_dummy_data", lambda n=200: (pd.DataFrame([[1]]), pd.Series([0])))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {}})

    module = types.SimpleNamespace(run_swarm_search=fake_swarm_search)
    monkeypatch.setitem(sys.modules, "swarm_sim", module)

    monkeypatch.setattr(sys, "argv", ["ml_trainer", "train", "regime", "--swarm"])

    ml_trainer.main()

    assert captured["params"].get("lr") == 0.1
