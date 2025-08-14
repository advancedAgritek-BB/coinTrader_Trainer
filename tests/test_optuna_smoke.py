import os
import sys

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


@pytest.mark.skipif(pytest.importorskip("optuna", reason="optuna not installed") is None, reason="no optuna")
def test_optuna_smoke(tmp_path):
    opt_mod = pytest.importorskip(
        "cointrainer.backtest.optuna_opt", reason="optuna optimizer unavailable"
    )
    optimize_optuna = opt_mod.optimize_optuna
    OptunaConfig = opt_mod.OptunaConfig
    # tiny synthetic dataset
    idx = pd.date_range("2025-01-01", periods=12000, freq="T", tz="UTC")
    price = 100 * (1 + 0.0005 * np.sin(np.arange(len(idx)) / 20)).cumprod()
    df = pd.DataFrame(
        {
            "open": price,
            "high": price * 1.0005,
            "low": price * 0.9995,
            "close": price,
            "volume": np.random.randint(1, 5, size=len(idx)),
            "trades": 1,
        },
        index=idx,
    )
    p = tmp_path / "SYN_1m.normalized.csv"
    df.to_csv(p)
    cfg = OptunaConfig(
        n_trials=2,
        n_folds=2,
        val_len=1000,
        gap=10,
        limit_rows=8000,
        device_type="cpu",
        max_bin=63,
        n_jobs=0,
    )
    res = optimize_optuna(p, "SYN", outdir=tmp_path, cfg=cfg)
    assert "best" in res

