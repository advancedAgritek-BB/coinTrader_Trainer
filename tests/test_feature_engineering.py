import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from feature_engineering import make_features


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

    result = make_features(df, rsi_period=10, vol_window=5, atr_period=4)
    for col in ['rsi10', 'volatility5', 'atr4']:
        assert col in result.columns


def test_make_features_raises_when_too_many_nans():
    df_small = pd.DataFrame({
        'ts': pd.date_range('2021-01-01', periods=5, freq='1T'),
        'price': np.arange(5),
    })
    with pytest.raises(ValueError):
        make_features(df_small)
