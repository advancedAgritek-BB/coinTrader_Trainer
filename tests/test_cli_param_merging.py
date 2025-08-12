import logging
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Provide a minimal supabase stub to satisfy imports during testing
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
sys.modules.setdefault("tenacity", types.SimpleNamespace(retry=lambda *a, **k: (lambda f: f), stop_after_attempt=lambda *a, **k: None, wait_exponential=lambda *a, **k: None))

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
    monkeypatch.setattr(ml_trainer, "check_clinfo_gpu", lambda: True)
    monkeypatch.setattr(ml_trainer, "verify_lightgbm_gpu", lambda p: True)

    ml_trainer.main()

    assert captured["gpu"]
    assert captured["params"].get("device") == "opencl"
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

    async def fake_swarm(start, end, *, table="ohlc_data"):
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

    async def fake_federated(start, end, *, table="ohlc_data", **kwargs):
        captured["start"] = start
        captured["end"] = end
        captured["params"] = kwargs.get("params_override", {}).copy()

        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

    monkeypatch.setattr(ml_trainer, "train_federated_regime", fake_federated)
    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        return object(), {}
    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(
        ml_trainer, "load_cfg", lambda p: {"federated_regime": {"objective": "binary"}}
    )
    monkeypatch.setattr(ml_trainer, "check_clinfo_gpu", lambda: True)
    monkeypatch.setattr(ml_trainer, "verify_lightgbm_gpu", lambda p: True)
    monkeypatch.setattr(ml_trainer, "verify_lightgbm_gpu", lambda p: True)
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
    assert captured["params"].get("device") == "opencl"


def test_true_federated_param_merge(monkeypatch):
    captured = {}

    def fake_start(start, end, *, config_path="cfg.yaml", params_override=None, table="ohlc_data"):
        captured["start"] = start
        captured["end"] = end
        captured["table"] = table
        captured["params"] = (params_override or {}).copy()

    module = types.SimpleNamespace(start_server=fake_start)
    monkeypatch.setitem(sys.modules, "federated_fl", module)
    monkeypatch.setattr(ml_trainer, "federated_fl", module, raising=False)
    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        return object(), {}
    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"federated_regime": {"objective": "binary"}})
    monkeypatch.setattr(ml_trainer, "check_clinfo_gpu", lambda: True)
    monkeypatch.setattr(ml_trainer, "verify_lightgbm_gpu", lambda p: True)

    argv = [
        "prog",
        "train",
        "regime",
        "--true-federated",
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
    assert captured["params"].get("device") == "opencl"


def test_true_federated_requires_range(monkeypatch):
    module = types.SimpleNamespace(launch=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "federated_fl", module)
    monkeypatch.setattr(ml_trainer, "federated_fl", module, raising=False)

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        return object(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"federated_regime": {"objective": "binary"}})

    argv = ["prog", "train", "regime", "--true-federated"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="--true-federated requires --start-ts and --end-ts"):
        ml_trainer.main()


def test_federated_missing_support_warns(monkeypatch, caplog):
    monkeypatch.setattr(ml_trainer, "train_federated_regime", None, raising=False)

    called = {}

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        called["train"] = True
        return object(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {}})
    monkeypatch.setattr(ml_trainer, "check_clinfo_gpu", lambda: True)
    monkeypatch.setattr(ml_trainer, "verify_lightgbm_gpu", lambda p: True)
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (pd.DataFrame({"f": [1]}), pd.Series([0])),
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
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with caplog.at_level(logging.WARNING):
        ml_trainer.main()

    assert called.get("train")
    assert any(
        "Federated training not supported" in r.message for r in caplog.records
    )


def test_true_federated_missing_flwr_warns(monkeypatch, caplog):
    module = types.SimpleNamespace(_HAVE_FLWR=False)
    monkeypatch.setitem(sys.modules, "federated_fl", module)
    monkeypatch.setattr(ml_trainer, "federated_fl", module, raising=False)

    called = {}

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        called["train"] = True
        return object(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {}})
    monkeypatch.setattr(ml_trainer, "check_clinfo_gpu", lambda: True)
    monkeypatch.setattr(ml_trainer, "verify_lightgbm_gpu", lambda p: True)
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (pd.DataFrame({"f": [1]}), pd.Series([0])),
    )

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

    assert not called.get("train")
    assert any(
        "True federated training requires 'flwr'" in r.message
        for r in caplog.records
    )

