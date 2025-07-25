import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ml_trainer


def _basic_dataset():
    return pd.DataFrame(np.random.normal(size=(4, 2))), pd.Series([0, 1, 0, 1])


def test_gpu_flag_merges_params(monkeypatch):
    captured = {}

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        captured["params"] = params.copy()
        captured["gpu"] = use_gpu

        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(
        ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {"objective": "binary"}}
    )
    monkeypatch.setattr(ml_trainer, "_make_dummy_data", lambda n=200: _basic_dataset())

    argv = [
        "prog",
        "train",
        "regime",
        "--use-gpu",
        "--gpu-platform-id",
        "1",
        "--gpu-device-id",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    ml_trainer.main()

    assert captured["gpu"]
    assert captured["params"].get("device_type") == "gpu"
    assert captured["params"].get("gpu_platform_id") == 1
    assert captured["params"].get("gpu_device_id") == 2


def test_swarm_merges_once(monkeypatch):
    captured = {"count": 0}

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        captured["params"] = params.copy()

        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

    async def fake_swarm(start, end, *, table="trade_logs"):
        captured["count"] += 1
        return {"lr": 0.1}

    module = types.SimpleNamespace(run_swarm_search=fake_swarm)
    monkeypatch.setitem(sys.modules, "swarm_sim", module)

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(
        ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {"objective": "binary"}}
    )
    monkeypatch.setattr(ml_trainer, "_make_dummy_data", lambda n=200: _basic_dataset())

    argv = ["prog", "train", "regime", "--swarm"]
    monkeypatch.setattr(sys, "argv", argv)

    ml_trainer.main()

    assert captured["count"] == 1
    assert captured["params"].get("lr") == 0.1


def test_federated_param_merge(monkeypatch):
    captured = {}

    def fake_federated(start, end, *, table="trade_logs", **kwargs):
        captured["start"] = start
        captured["end"] = end
        captured["params"] = kwargs.get("params_override", {}).copy()

        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

    monkeypatch.setattr(ml_trainer, "train_federated_regime", fake_federated)
    monkeypatch.setattr(
        ml_trainer, "load_cfg", lambda p: {"federated_regime": {"objective": "binary"}}
    )
    argv = [
        "prog",
        "train",
        "regime",
        "--federated",
        "--start-ts",
        "2021-01-01",
        "--end-ts",
        "2021-01-02",
        "--use-gpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    ml_trainer.main()

    assert captured["start"] == "2021-01-01"
    assert captured["end"] == "2021-01-02"
    assert captured["params"].get("device_type") == "gpu"
