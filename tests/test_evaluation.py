"""Tests for evaluation utilities."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluation import simulate_signal_pnl


def test_simulate_signal_pnl_metrics():
    df = pd.DataFrame({"returns": [0.01, -0.02, 0.03, -0.04]})
    preds = np.array([1, -1, -1, 1])

    metrics = simulate_signal_pnl(df, preds, costs=0.0, slippage=0.0)

    for key in ["sharpe_squared", "sharpe", "sortino"]:
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert np.isfinite(metrics[key])


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

