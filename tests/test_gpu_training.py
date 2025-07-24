import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trainers.regime_lgbm import train_regime_lgbm
import ml_trainer


class FakeBooster:
    best_iteration = 1
    def predict(self, data, num_iteration=None):
        return np.zeros(len(data))


def fake_federated(start, end, **kwargs):
    """Return a ``FakeBooster`` and empty metrics."""
    return FakeBooster(), {}
@pytest.fixture
def fake_federated(monkeypatch):
    """Patch ``train_federated_regime`` and record usage."""
    import ml_trainer

    called = {"used": False}

    def _fake(start, end, **kwargs):
        called["used"] = True

        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

    monkeypatch.setattr(ml_trainer, "train_federated_regime", _fake)
    return called


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
        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

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


def test_cli_federated_trainer_invoked(monkeypatch, fake_federated):
    import ml_trainer

    nonfed = {}

    def fake_train(*args, **kwargs):
        nonfed["used"] = True
        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

    monkeypatch.setattr(ml_trainer, "train_regime_lgbm", fake_train)
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (
            pd.DataFrame(np.random.normal(size=(10, 2))),
            pd.Series([0, 1] * 5),
        ),
    )
    monkeypatch.setattr(ml_trainer, "load_cfg", lambda p: {"federated_regime": {}})

    argv = [
        "ml_trainer",
        "train",
        "regime",
        "--federated",
        "--start-ts",
        "2021-01-01T00:00:00",
        "--end-ts",
        "2021-01-02T00:00:00",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    ml_trainer.main()

    assert fake_federated["used"] is True
    assert "used" not in nonfed


def test_cli_federated_flag(monkeypatch, fake_federated):
    import ml_trainer

    used = {}

    def fake_train(*args, **kwargs):
        used["called"] = True
        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

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

def test_cli_federated_trainer_invoked(monkeypatch):
    called = {}
    used = {}

    def capture_federated(start, end, **kwargs):
        called["args"] = (start, end)
        used["called"] = True
        return FakeBooster(), {}

    monkeypatch.setattr(ml_trainer, "train_federated_regime", capture_federated)
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

    assert used.get("called") is True
    assert called.get("args") == ("2021-01-01", "2021-01-02")

