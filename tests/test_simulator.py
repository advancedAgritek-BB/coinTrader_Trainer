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
    return pd.DataFrame({"pnl": np.array([0.1, -0.05, 0.2])})


backtest_strategies_mod.backtest = dummy_backtest
tools_mod.backtest_strategies = backtest_strategies_mod
sys.modules["tools"] = tools_mod
sys.modules["tools.backtest_strategies"] = backtest_strategies_mod

from utils.simulator import simulate_trades


def test_simulate_trades_returns_df_and_csv(tmp_path, monkeypatch):
    df = pd.DataFrame({"close": [1, 2, 3]})
    monkeypatch.chdir(tmp_path)

    trades_df = simulate_trades(df, [])

    pnl = np.array([0.1, -0.05, 0.2])
    expected_win_rate = (pnl > 0).mean()
    expected_sharpe = pnl.mean() / pd.Series(pnl).std() * np.sqrt(252)

    assert {"pnl", "win_rate", "sharpe"} <= set(trades_df.columns)
    assert trades_df["win_rate"].iloc[0] == pytest.approx(expected_win_rate)
    assert trades_df["sharpe"].iloc[0] == pytest.approx(expected_sharpe)
    assert (tmp_path / "simulated_trades.csv").exists()

