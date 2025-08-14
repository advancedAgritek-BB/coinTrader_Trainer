"""Tests for the backtesting wrapper."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest import run_backtest


def test_run_backtest_simple():
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "high": [1.0, 1.1, 1.2, 1.3, 1.4],
            "low": [1.0, 1.1, 1.2, 1.3, 1.4],
            "close": [1.0, 1.1, 1.2, 1.3, 1.4],
            "volume": [1, 1, 1, 1, 1],
        },
        index=pd.date_range("2021-01-01", periods=5, freq="D"),
    )
    signals = [1, 0, -1, 1, 0]

    stats = run_backtest(df, signals)

    assert "final_value" in stats
    assert isinstance(stats["final_value"], float)
    assert stats["final_value"] > 0


def test_run_backtest_requires_signals():
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

    with pytest.raises(ValueError, match="signals parameter is required"):
        run_backtest(df, None)

