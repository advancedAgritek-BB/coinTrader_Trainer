"""Verify that GPU-related parameters are passed to LightGBM."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trainers.regime_lgbm import train_regime_lgbm
import lightgbm as lgb


def test_gpu_params_passed_to_lightgbm(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(10, 3)))
    y = pd.Series([0, 1] * 5)

    captured = {}

    def fake_train(params, *args, **kwargs):
        captured.update(params)
        class FakeBooster:
            best_iteration = 1
            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))
        return FakeBooster()

    monkeypatch.setattr(lgb, "train", fake_train)

    params = {
        "objective": "binary",
        "num_boost_round": 5,
        "early_stopping_rounds": 2,
    }

    train_regime_lgbm(X, y, params, use_gpu=True)

    assert captured.get("device_type") == "gpu"


def test_cfg_loader_gpu_defaults(monkeypatch):
    import ml_trainer
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(10, 2)))
    y = pd.Series([0, 1] * 5)

    captured = {}

    def fake_train(params, *args, **kwargs):
        captured.update(params)
        class FakeBooster:
            best_iteration = 1
            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))
        return FakeBooster()

    monkeypatch.setattr(lgb, "train", fake_train)

    cfg = ml_trainer.load_cfg("cfg.yaml")
    params = cfg.get("regime_lgbm", {}).copy()
    params.pop("device_type", None)
    params.pop("gpu_platform_id", None)
    params.pop("gpu_device_id", None)

    train_regime_lgbm(X, y, params, use_gpu=True)

    assert captured.get("device_type") == "gpu"


def test_cli_gpu_overrides(monkeypatch):
    import ml_trainer

    captured = {}

    def fake_train(params, *args, **kwargs):
        captured.update(params)
        class FakeBooster:
            best_iteration = 1
            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))
        return FakeBooster()

    monkeypatch.setattr(lgb, "train", fake_train)
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (
            pd.DataFrame(np.random.normal(size=(10, 2))),
            pd.Series([0, 1] * 5),
        ),
    )

    original_load = ml_trainer.load_cfg

    def patched_load(path):
        cfg = original_load(path)
        cfg.get("regime_lgbm", {}).pop("device_type", None)
        return cfg

    monkeypatch.setattr(ml_trainer, "load_cfg", patched_load)

    argv = [
        "ml_trainer",
        "train",
        "regime",
        "--cfg",
        "cfg.yaml",
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

def test_cli_federated_flag(monkeypatch):
    import ml_trainer

    called = {}

    def fake_federated(*args, **kwargs):
        called["federated"] = True
        return (lambda df: np.zeros(len(df))), {}

    monkeypatch.setattr(ml_trainer, "train_federated_regime", fake_federated)
    def fake_train(*args, **kwargs):
        called["used"] = True
        class FakeBooster:
            best_iteration = 1
            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))
        return FakeBooster(), {"accuracy": 0.0}

    monkeypatch.setattr(ml_trainer, "train_regime_lgbm", fake_train)
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (
            pd.DataFrame(np.random.normal(size=(10, 2))),
            pd.Series([0, 1] * 5),
        ),
    )
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {}})
 
    argv = [
        "ml_trainer",
        "train",
        "regime",
        "--cfg",
        "cfg.yaml",
        "--federated",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    ml_trainer.main()

    assert called.get("federated")
    assert not called.get("used", False)
