import sys
import types
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    }

    assert expected_cols.issubset(result.columns)
    assert not result.isna().any().any()


def test_make_features_gpu_uses_opencl(monkeypatch):
    executed = {"kernel": False}

    class FakeKernel:
        def __call__(self, queue, global_size, local_size, *args):
            executed["kernel"] = True

    class FakeProgram:
        def __init__(self, ctx, src):
            pass

        def build(self):
            return self

        @property
        def compute_features(self):
            return FakeKernel()

    class FakePlatform:
        name = "AMD"

        def get_devices(self):
            return ["gpu"]

    class FakeQueue:
        def finish(self):
            pass

    def fake_buffer(ctx, flags, hostbuf=None, size=None):
        return types.SimpleNamespace(
            buf=np.empty(0) if hostbuf is None else hostbuf,
            nbytes=size if size is not None else getattr(hostbuf, "nbytes", 0)
        )

    fake_cl = types.SimpleNamespace(
        get_platforms=lambda: [FakePlatform()],
        Context=lambda devices=None: "ctx",
        CommandQueue=lambda ctx: FakeQueue(),
        mem_flags=types.SimpleNamespace(READ_ONLY=1, COPY_HOST_PTR=2, WRITE_ONLY=4),
        Buffer=fake_buffer,
        Program=FakeProgram,
        enqueue_copy=lambda queue, dest, src: None,
    )

    monkeypatch.setitem(sys.modules, "pyopencl", fake_cl)

    df = pd.DataFrame(
        {
            "ts": range(6),
            "price": [1, 2, 3, 4, 5, 6],
            "high": [1, 2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4, 5],
        }
    )

    make_features(df, use_gpu=True, ema_short_period=2, ema_long_period=3)

    assert executed["kernel"]


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
