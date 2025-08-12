import builtins
import logging
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

sys.modules.setdefault(
    "supabase",
    types.SimpleNamespace(
        create_client=lambda *a, **k: object(),
        Client=object,
        SupabaseException=Exception,
    ),
)
_pg_exc = types.SimpleNamespace(APIError=Exception)
sys.modules.setdefault("postgrest", types.SimpleNamespace(exceptions=_pg_exc))
sys.modules.setdefault("postgrest.exceptions", _pg_exc)
sys.modules.setdefault(
    "tenacity",
    types.SimpleNamespace(
        retry=lambda *a, **k: (lambda f: f),
        stop_after_attempt=lambda *a, **k: None,
        wait_exponential=lambda *a, **k: None,
    ),
)
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("redis", types.ModuleType("redis"))
sys.modules.setdefault("requests", types.ModuleType("requests"))
bt_mod = types.ModuleType("backtrader")
bt_mod.Strategy = object
class DummyCerebro:
    def __init__(self):
        pass
bt_mod.Cerebro = DummyCerebro
bt_mod.feeds = types.SimpleNamespace(PandasData=object)
sys.modules["backtrader"] = bt_mod
sk_utils = types.ModuleType("sklearn.utils")
sk_utils.resample = lambda *a, **k: None
sk_utils.shuffle = lambda *a, **k: None
sk_mod = types.ModuleType("sklearn")
metrics_mod = types.ModuleType("sklearn.metrics")
metrics_mod.accuracy_score = lambda *a, **k: 0.0
metrics_mod.f1_score = lambda *a, **k: 0.0
metrics_mod.precision_score = lambda *a, **k: 0.0
metrics_mod.recall_score = lambda *a, **k: 0.0
sys.modules.setdefault("sklearn", sk_mod)
sys.modules.setdefault("sklearn.utils", sk_utils)
sys.modules.setdefault("sklearn.metrics", metrics_mod)
ms_mod = types.ModuleType("sklearn.model_selection")
ms_mod.StratifiedKFold = object
sys.modules.setdefault("sklearn.model_selection", ms_mod)
sys.modules.setdefault("numba", types.ModuleType("numba"))
sys.modules["numba"].njit = lambda f=None, **k: (lambda *a, **kw: None) if f is None else f
sys.modules.setdefault("joblib", types.ModuleType("joblib"))
sys.modules.setdefault("lightgbm", types.ModuleType("lightgbm"))
sys.modules["lightgbm"].Booster = object
sys.modules.setdefault("optuna", types.ModuleType("optuna"))

import ml_trainer


def _basic_dataset():
    return pd.DataFrame(np.random.normal(size=(4, 2))), pd.Series([0, 1, 0, 1])


def test_swarm_missing_logs_warning(monkeypatch, caplog):
    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        return object(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {"objective": "binary"}})
    monkeypatch.setattr(ml_trainer, "_make_dummy_data", lambda n=200: _basic_dataset())

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "swarm_sim":
            raise ImportError("missing swarm")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    argv = ["prog", "train", "regime", "--swarm"]
    monkeypatch.setattr(sys, "argv", argv)

    with caplog.at_level(logging.WARNING):
        ml_trainer.main()

    assert "Swarm optimization unavailable" in caplog.text


def test_optuna_missing_logs_warning(monkeypatch, caplog):
    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        return object(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {"objective": "binary"}})
    monkeypatch.setattr(ml_trainer, "_make_dummy_data", lambda n=200: _basic_dataset())

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"optuna_search", "optuna_optimizer"}:
            raise ImportError("missing optuna")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    argv = ["prog", "train", "regime", "--optuna"]
    monkeypatch.setattr(sys, "argv", argv)

    with caplog.at_level(logging.WARNING):
        ml_trainer.main()

    assert "optimization unavailable" in caplog.text


def test_true_federated_missing_logs_warning(monkeypatch, caplog):
    fake_fl = types.SimpleNamespace(_HAVE_FLWR=False)
    monkeypatch.setitem(sys.modules, "federated_fl", fake_fl)
    monkeypatch.setattr(ml_trainer, "federated_fl", fake_fl)

    argv = [
        "prog",
        "train",
        "regime",
        "--true-federated",
        "--start-ts",
        "2021-01-01",
        "--end-ts",
        "2021-01-02",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with caplog.at_level(logging.WARNING):
        ml_trainer.main()

    assert "requires the 'flwr'" in caplog.text
