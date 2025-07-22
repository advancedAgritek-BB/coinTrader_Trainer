"""Unit tests for the LightGBM regime trainer."""

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import Booster

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from trainers.regime_lgbm import train_regime_lgbm


def test_train_regime_lgbm_returns_model_and_metrics():
    # deterministic synthetic data
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series([0, 1] * 10)

    params = {
        "objective": "binary",
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
