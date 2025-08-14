import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cointrainer.backtest.optimize import optimize_grid
from cointrainer.backtest import run as bt_run


class _DummyModel:
    def predict_proba(self, X):
        # return uniform probabilities for three classes
        return np.ones((len(X), 3)) / 3


def test_grid_optimizer_runs(tmp_path, monkeypatch):
    # small synthetic normalized csv
    idx = pd.date_range("2025-01-01", periods=5000, freq="T", tz="UTC")
    price = 100 * (1 + 0.0005 * np.sin(np.arange(len(idx)) / 20)).cumprod()
    df = pd.DataFrame(
        {
            "open": price,
            "high": price * 1.0005,
            "low": price * 0.9995,
            "close": price,
            "volume": np.random.randint(1, 5, size=len(idx)),
            "trades": np.random.randint(1, 3, size=len(idx)),
        },
        index=idx,
    )
    p = tmp_path / "SYN_1m.normalized.csv"
    df.to_csv(p, index=True)

    # Patch model loading to avoid joblib dependency on a real model file
    monkeypatch.setattr(bt_run, "load_model_local", lambda path: _DummyModel())

    res = optimize_grid(
        csv_path=p,
        symbol="SYN",
        horizons=[15],
        holds=[0.0015],
        open_thrs=[0.55],
        position_modes=["gated"],
        fee_bps=2.0,
        slip_bps=0.0,
        device_type="cpu",
        max_bin=63,
        n_jobs=0,
        limit_rows=4000,
        outdir=tmp_path,
    )
    assert "best" in res and "leaderboard" in res

