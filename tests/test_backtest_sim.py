import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from cointrainer.backtest.sim import simulate


def test_sim_basic():
    idx = pd.date_range("2025-01-01", periods=100, freq="T", tz="UTC")
    price = pd.Series(100.0 * (1.0 + 0.0005*np.sin(np.arange(100)/10)).cumprod(), index=idx)
    pos = np.where(np.arange(100) % 10 < 5, 1.0, 0.0)  # long half the time
    res = simulate(price, pos, fee_bps=2.0, slip_bps=0.0)
    assert "stats" in res and "final_equity" in res["stats"]
