"""Unit tests for the LightGBM regime trainer."""

import os
import sys
import types

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Booster

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trainers.regime_lgbm import train_regime_lgbm


def test_train_regime_lgbm_returns_model_and_metrics():
    # deterministic synthetic data
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series([-1, 0, 1] * 10)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "verbose": -1,
        "num_boost_round": 10,
        "early_stopping_rounds": 5,
    }

    model, metrics = train_regime_lgbm(X, y, params, use_gpu=False)

    assert isinstance(model, Booster)
    assert isinstance(metrics, dict)
    for key in ["accuracy", "f1", "precision_long", "recall_long"]:
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_train_regime_lgbm_with_tuning():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series([-1, 0, 1] * 10)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "verbose": -1,
        "num_boost_round": 10,
        "early_stopping_rounds": 5,
    }

    model, metrics = train_regime_lgbm(
        X, y, params, use_gpu=False, tune=True, n_trials=2
    )

    assert isinstance(model, Booster)
    assert isinstance(metrics, dict)


def _fake_booster():
    class FakeBooster(Booster):
        def __init__(self, *args, **kwargs):
            pass

        best_iteration = 1

        def predict(self, data, num_iteration=None):
            return np.zeros((len(data), 3))

    return FakeBooster()


def test_label_encoding(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(15, 3)))
    y = pd.Series([-1, 0, 1] * 5)

    captured = {}

    def fake_train(params, train_set, *args, **kwargs):
        captured[0] = params
        captured["labels"] = train_set.get_label()
        captured[0] = params
        return _fake_booster()

    monkeypatch.setattr(lgb, "train", fake_train)

    params = {"objective": "multiclass", "num_class": 3, "num_boost_round": 5, "early_stopping_rounds": 2}
    train_regime_lgbm(X, y, params, use_gpu=False)

    assert set(captured["labels"]) == {0, 1, 2}
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    expected = neg / pos
    assert "scale_pos_weight" not in params or params["scale_pos_weight"] == expected


def test_optuna_tuning_sets_learning_rate(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 3)))
    y = pd.Series([-1, 0, 1] * 10)

    best_lr = 0.05
    calls = {}

    class FakeStudy:
        best_params = {"learning_rate": best_lr}

        def optimize(self, obj, n_trials=10):
            calls["optimize"] = True

    def fake_create_study(direction="minimize"):
        calls["create"] = True
        return FakeStudy()

    fake_optuna = types.SimpleNamespace(create_study=fake_create_study)
    monkeypatch.setitem(train_regime_lgbm.__globals__, "optuna", fake_optuna)
    try:
        monkeypatch.setattr(train_regime_lgbm.__globals__, "optuna", fake_optuna)
    except AttributeError:
        pass
    monkeypatch.setattr(lgb, "train", lambda *a, **k: _fake_booster())

    params = {"objective": "multiclass", "num_class": 3, "num_boost_round": 5, "tune_learning_rate": True}
    train_regime_lgbm(X, y, params, use_gpu=False)

    assert calls.get("create")
    assert params["learning_rate"] == best_lr


def test_train_regime_lgbm_tune_and_lr(monkeypatch):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(30, 3)))
    y = pd.Series([-1, 0, 1] * 10)

    class FakeStudy:
        best_params = {"learning_rate": 0.1}

        def optimize(self, obj, n_trials=2):
            pass

    fake_optuna = types.SimpleNamespace(
        create_study=lambda direction="minimize": FakeStudy()
    )
    try:
        monkeypatch.setattr(train_regime_lgbm.__globals__, "optuna", fake_optuna)
    except AttributeError:
        pass
    monkeypatch.setattr(lgb, "train", lambda *a, **k: _fake_booster())

    params = {"objective": "multiclass", "num_class": 3, "num_boost_round": 5, "tune_learning_rate": True}
    model, metrics = train_regime_lgbm(
        X, y, params, use_gpu=False, tune=True, n_trials=2
    )
    assert isinstance(model, Booster)


def test_model_registry_upload_called(monkeypatch, registry_with_dummy):
    reg, _ = registry_with_dummy

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 3)))
    y = pd.Series([-1, 0, 1] * 10)

    uploaded = {}

    def fake_upload(model, name, metrics):
        uploaded["model"] = model
        uploaded["metrics"] = metrics

    monkeypatch.setattr(lgb, "train", lambda *a, **k: _fake_booster())
    monkeypatch.setattr(reg, "upload", fake_upload, raising=False)

    params = {"objective": "multiclass", "num_class": 3, "num_boost_round": 5}
    model, metrics = train_regime_lgbm(X, y, params, use_gpu=False, registry=reg)

    assert uploaded["model"] is model
    assert uploaded["metrics"] == metrics
