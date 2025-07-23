import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trainers.regime_lgbm import train_regime_lgbm
import ml_trainer


class FakeBooster:
    best_iteration = 1
    def predict(self, data, num_iteration=None):
        return np.zeros(len(data))


def test_gpu_params_passed_to_lightgbm(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(10, 3)))
    y = pd.Series([0, 1] * 5)

    captured = {}

    def fake_train(params, *args, **kwargs):
        captured.update(params)
        return FakeBooster()

    monkeypatch.setattr(lgb, "train", fake_train)

    params = {"objective": "binary", "num_boost_round": 5, "early_stopping_rounds": 2}

    train_regime_lgbm(X, y, params, use_gpu=True)

    assert captured.get("device_type") == "gpu"


def test_cli_gpu_overrides(monkeypatch):
    captured = {}

    def fake_train(X, y, params, use_gpu=False):
        captured.update(params)
        return FakeBooster(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {"objective": "binary"}})
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (pd.DataFrame(np.random.normal(size=(10, 2))), pd.Series([0, 1] * 5)),
    )

    argv = [
        "ml_trainer",
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

    assert captured.get("device_type") == "gpu"
    assert captured.get("gpu_platform_id") == 1
    assert captured.get("gpu_device_id") == 2


def test_cli_federated_trainer_invoked(monkeypatch):
    called = {}

    def fake_federated(start, end, **kwargs):
        called["args"] = (start, end)
        return FakeBooster(), {}

    monkeypatch.setattr(ml_trainer, "train_federated_regime", fake_federated)
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"federated_regime": {"objective": "binary"}})
    argv = [
        "ml_trainer",
        "train",
        "regime",
        "--federated",
        "--start-ts",
        "2021-01-01",
        "--end-ts",
        "2021-01-02",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    ml_trainer.main()

    assert called.get("args") == ("2021-01-01", "2021-01-02")

