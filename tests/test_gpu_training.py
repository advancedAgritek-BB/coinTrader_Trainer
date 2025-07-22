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
        "device_type": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }

    train_regime_lgbm(X, y, params)

    assert captured.get("device_type") == "gpu"
