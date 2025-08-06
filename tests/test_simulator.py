"""Tests for the simulator utility."""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Create a dummy tools.backtest_strategies module
tools_mod = types.ModuleType("tools")
backtest_strategies_mod = types.ModuleType("tools.backtest_strategies")


def dummy_backtest(df, strategies):
    return {"pnl": np.array([0.1, -0.05, 0.2])}


backtest_strategies_mod.backtest = dummy_backtest
tools_mod.backtest_strategies = backtest_strategies_mod
sys.modules["tools"] = tools_mod
sys.modules["tools.backtest_strategies"] = backtest_strategies_mod

from utils.simulator import simulate


def test_simulate_returns_metrics():
    df = pd.DataFrame({"close": [1, 2, 3]})
    metrics = simulate(df, [])
    pnl = np.array([0.1, -0.05, 0.2])
    expected_win_rate = (pnl > 0).mean()
    expected_sharpe = pnl.mean() / pnl.std(ddof=0) * np.sqrt(252)
    assert metrics["win_rate"] == pytest.approx(expected_win_rate)
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)
