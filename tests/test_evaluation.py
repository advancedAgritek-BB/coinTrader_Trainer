"""Tests for evaluation utilities."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluation import simulate_signal_pnl, full_strategy_eval


def test_simulate_signal_pnl_metrics():
    df = pd.DataFrame({"returns": [0.01, -0.02, 0.03, -0.04]})
    preds = np.array([1, -1, -1, 1])

    metrics = simulate_signal_pnl(df, preds, costs=0.0, slippage=0.0)

    for key in [
        "sharpe_squared",
        "sharpe",
        "sortino",
        "max_drawdown",
        "win_rate",
        "calmar_ratio",
        "profit_factor",
    ]:
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert np.isfinite(metrics[key])

    # ensure metric names are unique
    assert len(metrics) == 7


def test_simulate_signal_pnl_zero_variance():
    df = pd.DataFrame({"returns": [0.0, 0.0, 0.0, 0.0]})
    preds = np.array([1, -1, 1, -1])

    metrics = simulate_signal_pnl(df, preds, costs=0.0, slippage=0.0)

    assert metrics["sharpe"] == 0.0
    assert metrics["sortino"] == 0.0


def test_simulate_signal_pnl_slippage_reduces_returns():
    df = pd.DataFrame({"returns": [0.02, -0.02, 0.02, -0.02]})
    preds = np.array([1, -1, 1, -1])

    no_slip = simulate_signal_pnl(df, preds, costs=0.0, slippage=0.0)
    with_slip = simulate_signal_pnl(df, preds, costs=0.0, slippage=0.05)

    assert with_slip["sharpe"] <= no_slip["sharpe"]


def test_full_strategy_eval_returns_value():
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2],
            "volume": [1, 1, 1],
        },
        index=pd.date_range("2021-01-01", periods=3, freq="D"),
    )
    preds = np.array([1, 0, -1])

    metrics = full_strategy_eval(df, preds, slippage=0.0, costs=0.0)

    assert "final_portfolio_value" in metrics
    assert isinstance(metrics["final_portfolio_value"], float)
    assert "calmar_ratio" in metrics
    assert isinstance(metrics["calmar_ratio"], float)

