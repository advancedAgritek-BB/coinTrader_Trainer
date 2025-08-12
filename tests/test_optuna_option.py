import builtins
import logging
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ml_trainer


def test_cli_optuna_merges_params_once(monkeypatch):
    captured = {"count": 0}

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        captured["params"] = params.copy()

        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(
        ml_trainer,
        "load_cfg",
        lambda path: {
            "regime_lgbm": {"objective": "binary"},
            "default_window_days": 7,
            "optuna": {"foo": "bar"},
        },
    )
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (
            pd.DataFrame(np.random.normal(size=(4, 2))),
            pd.Series([0, 1, 0, 1]),
        ),
    )

    def fake_run(window, *, table="ohlc_data", foo=None):
        captured["count"] += 1
        captured["window"] = window
        captured["table"] = table
        captured["foo"] = foo
        return {"learning_rate": 0.1}

    module = types.SimpleNamespace(run_optuna_search=fake_run)
    monkeypatch.setitem(sys.modules, "optuna_optimizer", module)

    argv = ["ml_trainer", "train", "regime", "--optuna"]
    monkeypatch.setattr(sys, "argv", argv)

    ml_trainer.main()

    assert captured["count"] == 1
    assert captured["params"].get("learning_rate") == 0.1


def test_cli_optuna_missing_module_warns(monkeypatch, caplog):
    called = {}

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        called["train"] = True
        return object(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {}})
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (pd.DataFrame({"f": [1]}), pd.Series([0])),
    )
    monkeypatch.setattr(ml_trainer, "check_clinfo_gpu", lambda: True)
    monkeypatch.setattr(ml_trainer, "verify_lightgbm_gpu", lambda p: True)
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("optuna_search", "optuna_optimizer"):
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    argv = ["prog", "train", "regime", "--optuna"]
    monkeypatch.setattr(sys, "argv", argv)

    with caplog.at_level(logging.WARNING):
        ml_trainer.main()

    assert called.get("train")
    assert any(
        "Optuna optimisation unavailable" in r.message for r in caplog.records
    )
