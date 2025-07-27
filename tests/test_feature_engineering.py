import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineering import make_features


def test_make_features_interpolation_and_columns():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=6, freq="D"),
            "price": [1.0, 1.1, np.nan, 1.3, 1.2, 1.4],
            "high": [1.1, 1.2, 1.3, np.nan, 1.25, 1.5],
            "low": [0.9, 1.0, 1.1, 1.15, np.nan, 1.3],
            "volume": [10, 11, 12, 13, 14, 15],
        }
    )

    result = make_features(
        df,
        ema_short_period=2,
        ema_long_period=3,
        rsi_period=5,
        volatility_window=2,
        atr_window=2,
    )

    expected_cols = {
        "ema_short",
        "ema_long",
        "macd",
        "rsi5",
        "volatility2",
        "atr2",
        "bol_upper",
        "bol_mid",
        "bol_lower",
        "momentum_10",
        "adx_14",
        "obv",
    }

    assert expected_cols.issubset(result.columns)
    assert not result.isna().any().any()


def test_make_features_gpu_uses_cudf(monkeypatch):
    calls = {"from": False, "to": False}

    class FakeDF(pd.DataFrame):
        def to_pandas(self):
            calls["to"] = True
            return pd.DataFrame(self)

    module = types.ModuleType("cudf")

    def from_pandas(df):
        calls["from"] = True
        return FakeDF(df)

    module.from_pandas = from_pandas
    monkeypatch.setitem(sys.modules, "cudf", module)

    df = pd.DataFrame(
        {
            "ts": range(6),
            "price": [1, 2, 3, 4, 5, 6],
            "high": [1, 2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4, 5],
            "volume": [1, 1, 1, 1, 1, 1],
        }
    )

    make_features(df, use_gpu=True, ema_short_period=2, ema_long_period=3)

    assert calls["from"] and calls["to"]


def test_make_features_gpu_generates_columns(monkeypatch):
    module = types.ModuleType("cudf")

    class FakeDF(pd.DataFrame):
        def to_pandas(self):
            return pd.DataFrame(self)

    def from_pandas(df):
        return FakeDF(df)

    module.from_pandas = from_pandas
    module.concat = pd.concat
    module.to_datetime = pd.to_datetime
    monkeypatch.setitem(sys.modules, "cudf", module)

    df = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=6, freq="D"),
            "price": [1.0, 1.1, 1.2, 1.3, 1.2, 1.4],
            "high": [1.1, 1.2, 1.3, 1.4, 1.25, 1.5],
            "low": [0.9, 1.0, 1.1, 1.15, 1.0, 1.3],
            "volume": [1, 1, 1, 1, 1, 1],
        }
    )

    result = make_features(
        df,
        ema_short_period=2,
        ema_long_period=3,
        rsi_period=5,
        volatility_window=2,
        atr_window=2,
        use_gpu=True,
    )

    expected_cols = {
        "ema_short",
        "ema_long",
        "macd",
        "rsi5",
        "volatility2",
        "atr2",
        "bol_upper",
        "bol_mid",
        "bol_lower",
        "momentum_10",
        "adx_14",
        "obv",
    }

    assert expected_cols.issubset(result.columns)


def test_make_features_adds_columns_and_handles_params(caplog):
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=300, freq="1T"),
            "price": np.linspace(100, 399, 300),
            "high": np.linspace(101, 400, 300),
            "low": np.linspace(99, 398, 300),
            "volume": np.ones(300),
        }
    )

    with caplog.at_level(logging.INFO):
        result = make_features(df)
    assert any("make_features took" in r.message for r in caplog.records)
    for col in [
        "log_ret",
        "ema_short",
        "ema_long",
        "macd",
        "rsi14",
        "volatility20",
        "atr3",
        "bol_upper",
        "bol_mid",
        "bol_lower",
        "momentum_10",
        "adx_14",
        "obv",
    ]:
        assert col in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["ts"])

    result = make_features(df, rsi_period=10, volatility_window=5, atr_window=4)
    for col in ["rsi10", "volatility5", "atr4"]:
        assert col in result.columns


def test_make_features_creates_multiclass_target():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2022-01-01", periods=4, freq="D"),
            "price": [1.0, 1.1, 1.0, 1.2],
            "high": [1.1, 1.2, 1.1, 1.3],
            "low": [0.9, 1.0, 0.9, 1.1],
        }
    )

    result = make_features(df)
    assert "target" in result.columns
    assert set(result["target"].unique()).issubset({-1, 0, 1})


def test_make_features_raises_when_too_many_nans():
    df_small = pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=5, freq="1T"),
            "price": np.arange(5),
        }
    )
    with pytest.raises(ValueError):
        make_features(df_small)


def test_make_features_generates_target_when_missing():
    prices = [1, 1.02, 0.98, 1.05, 1.06, 1.07]
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=len(prices), freq="D"),
            "price": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
        }
    )

    result = make_features(
        df,
        ema_short_period=1,
        ema_long_period=1,
        rsi_period=2,
        volatility_window=2,
        atr_window=2,
    )

    returns = result["price"].pct_change().shift(-1)
    expected = np.where(
        returns > 0.01,
        1,
        np.where(returns < -0.01, -1, 0),
    )
    expected = pd.Series(expected, index=result.index, name="target").fillna(0)

    pd.testing.assert_series_equal(result["target"], expected)
