import os
import sys
import types
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from feature_engineering import make_features


def test_make_features_interpolation_and_columns():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=6, freq="D"),
            "price": [1.0, 1.1, np.nan, 1.3, 1.2, 1.4],
            "high": [1.1, 1.2, 1.3, np.nan, 1.25, 1.5],
            "low": [0.9, 1.0, 1.1, 1.15, np.nan, 1.3],
        }
    )

    result = make_features(
        df,
        ema_short=2,
        ema_long=3,
        rsi_period=5,
        vol_window=2,
        atr_period=2,
    )

    expected_cols = {
        "ema_short",
        "ema_long",
        "macd",
        "rsi5",
        "volatility2",
        "atr2",
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
        }
    )

    make_features(df, use_gpu=True, ema_short=2, ema_long=3)

    assert calls["from"] and calls["to"]


def test_make_features_adds_columns_and_handles_params(capsys):
    df = pd.DataFrame({
        'ts': pd.date_range('2021-01-01', periods=300, freq='1T'),
        'price': np.linspace(100, 399, 300),
        'high': np.linspace(101, 400, 300),
        'low': np.linspace(99, 398, 300),
    })

    result = make_features(df, log_time=True)
    captured = capsys.readouterr().out
    assert 'feature generation took' in captured
    for col in ['log_ret', 'ema_short', 'ema_long', 'macd', 'rsi14', 'volatility20', 'atr3']:
        assert col in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result['ts'])

    result = make_features(df, rsi_period=10, volatility_window=5, atr_window=4)
    for col in ['rsi10', 'volatility5', 'atr4']:
        assert col in result.columns


def test_make_features_raises_when_too_many_nans():
    df_small = pd.DataFrame({
        'ts': pd.date_range('2021-01-01', periods=5, freq='1T'),
        'price': np.arange(5),
    })
    with pytest.raises(ValueError):
        make_features(df_small)
