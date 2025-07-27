import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import fakeredis
import logging
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import feature_engineering
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


def test_make_features_gpu_matches_cpu(monkeypatch):
    monkeypatch.setenv("ROCM_PATH", "1")
    monkeypatch.setattr(feature_engineering.platform, "system", lambda: "Windows")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    orig_compute = feature_engineering._compute_features_pandas
    def _no_numba(*args, **kwargs):
        return orig_compute(*args[:10], use_numba=False)

    monkeypatch.setattr(feature_engineering, "_rsi_nb", lambda arr, period=14: feature_engineering._rsi(pd.Series(arr), period).to_numpy())
    monkeypatch.setattr(feature_engineering, "_atr_nb", lambda h, l, c, period=14: feature_engineering._atr(pd.DataFrame({"high": h, "low": l, "price": c}), period).to_numpy())
    monkeypatch.setattr(feature_engineering, "_adx_nb", lambda h, l, c, period=14: feature_engineering._adx(pd.DataFrame({"high": h, "low": l, "price": c}), period).to_numpy())
    monkeypatch.setattr(feature_engineering, "_obv_nb", lambda price, volume: feature_engineering._obv(pd.DataFrame({"price": price, "volume": volume})).to_numpy())

    monkeypatch.setattr(feature_engineering, "_compute_features_pandas", _no_numba)
    jnp = types.ModuleType("jax.numpy")
    jnp.asarray = np.asarray
    jax_mod = types.ModuleType("jax")
    jax_mod.numpy = jnp
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)
    monkeypatch.setattr(feature_engineering, "jnp", jnp, raising=False)

def test_make_features_gpu_uses_jax(monkeypatch):
    calls = {"asarray": False}
    monkeypatch.setenv("ROCM_PATH", "1")
    monkeypatch.setattr(feature_engineering.platform, "system", lambda: "Windows")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    orig_compute = feature_engineering._compute_features_pandas
    def _no_numba2(*args, **kwargs):
        return orig_compute(*args[:10], use_numba=False)

    monkeypatch.setattr(feature_engineering, "_rsi_nb", lambda arr, period=14: feature_engineering._rsi(pd.Series(arr), period).to_numpy())
    monkeypatch.setattr(feature_engineering, "_atr_nb", lambda h, l, c, period=14: feature_engineering._atr(pd.DataFrame({"high": h, "low": l, "price": c}), period).to_numpy())
    monkeypatch.setattr(feature_engineering, "_adx_nb", lambda h, l, c, period=14: feature_engineering._adx(pd.DataFrame({"high": h, "low": l, "price": c}), period).to_numpy())
    monkeypatch.setattr(feature_engineering, "_obv_nb", lambda price, volume: feature_engineering._obv(pd.DataFrame({"price": price, "volume": volume})).to_numpy())

    monkeypatch.setattr(feature_engineering, "_compute_features_pandas", _no_numba2)
    monkeypatch.setattr(feature_engineering, "_atr_nb", lambda h, l, c, period=14: feature_engineering._atr(pd.DataFrame({"high": h, "low": l, "price": c}), period).to_numpy())
    monkeypatch.setattr(feature_engineering, "_adx_nb", lambda h, l, c, period=14: feature_engineering._adx(pd.DataFrame({"high": h, "low": l, "price": c}), period).to_numpy())
    monkeypatch.setattr(feature_engineering, "_obv_nb", lambda price, volume: feature_engineering._obv(pd.DataFrame({"price": price, "volume": volume})).to_numpy())

    jnp = types.ModuleType("jax.numpy")

    def asarray(x):
        calls["asarray"] = True
        return np.asarray(x)

    jnp.asarray = asarray
    jax_mod = types.ModuleType("jax")
    jax_mod.numpy = jnp
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)
    monkeypatch.setattr(feature_engineering, "jnp", jnp, raising=False)

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

    assert calls["asarray"]


def test_make_features_gpu_generates_columns(monkeypatch):
    jnp = types.ModuleType("jax.numpy")
    jnp.asarray = np.asarray
    jax_mod = types.ModuleType("jax")
    jax_mod.numpy = jnp
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)
    monkeypatch.setattr(feature_engineering, "jnp", jnp, raising=False)

def test_make_features_gpu_matches_cpu(monkeypatch):
    monkeypatch.setenv("ROCM_PATH", "1")
    monkeypatch.setattr(feature_engineering.platform, "system", lambda: "Windows")
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=6, freq="D"),
            "price": [1.0, 1.1, 1.2, 1.3, 1.2, 1.4],
            "high": [1.1, 1.2, 1.3, 1.4, 1.25, 1.5],
            "low": [0.9, 1.0, 1.1, 1.15, 1.0, 1.3],
            "volume": [1, 1, 1, 1, 1, 1],
        }
    )

    cpu = make_features(
        df,
        ema_short_period=2,
        ema_long_period=3,
        rsi_period=5,
        volatility_window=2,
        atr_window=2,
        use_gpu=False,
    )
    gpu = make_features(
        df,
        ema_short_period=2,
        ema_long_period=3,
        rsi_period=5,
        volatility_window=2,
        atr_window=2,
        use_gpu=True,
    )

    pd.testing.assert_frame_equal(cpu, gpu)


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


def test_make_features_handles_missing_high_low():
    df_small = pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=30, freq="1T"),
            "price": np.arange(30),
        }
    )
    make_features(df_small)


def test_make_features_handles_insufficient_rows():
    df_short = pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=5, freq="1T"),
            "price": np.arange(5),
            "high": np.arange(5),
            "low": np.arange(5),
            "volume": np.arange(5),
        }
    )

    make_features(df_short)


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


def test_make_features_modin_roundtrip(monkeypatch):
    calls = {"construct": False}

    module = types.ModuleType("modin.pandas")

    class FakeDF(pd.DataFrame):
        def __init__(self, *args, **kwargs):
            calls["construct"] = True
            super().__init__(*args, **kwargs)

        def to_pandas(self):
            return pd.DataFrame(self)

    module.DataFrame = FakeDF
    module.concat = pd.concat
    module.Series = pd.Series
    module.to_datetime = pd.to_datetime
    monkeypatch.setitem(sys.modules, "modin.pandas", module)

    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=3, freq="D"),
        "price": [1.0, 1.1, 1.2],
        "high": [1.0, 1.1, 1.2],
        "low": [0.9, 1.0, 1.1],
    })

    result = make_features(df, use_modin=True)
    assert calls["construct"]
    assert isinstance(result, pd.DataFrame)


def test_make_features_dask_roundtrip(monkeypatch):
    calls = {"from_pandas": False}

    module = types.ModuleType("dask.dataframe")

    class FakeDaskDF:
        def __init__(self, pdf):
            self.pdf = pdf

        def map_partitions(self, func):
            self.func = func
            return self

        def compute(self):
            return self.func(self.pdf)

    def from_pandas(df, npartitions=1):
        calls["from_pandas"] = True
        return FakeDaskDF(df)

    module.from_pandas = from_pandas
    module.Index = pd.Index
    dask_mod = types.ModuleType("dask")
    dask_mod.dataframe = module
    monkeypatch.setitem(sys.modules, "dask", dask_mod)
    monkeypatch.setitem(sys.modules, "dask.dataframe", module)

    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=3, freq="D"),
        "price": [1.0, 1.1, 1.2],
        "high": [1.0, 1.1, 1.2],
        "low": [0.9, 1.0, 1.1],
    })

    result = make_features(df, use_dask=True)
    assert calls["from_pandas"]
    assert isinstance(result, pd.DataFrame)
def test_make_features_warns_when_overwriting_target(caplog):
    prices = [1, 1.02, 0.98, 1.05, 1.06, 1.07]
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=len(prices), freq="D"),
            "price": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "target": [1, np.nan, -1, 0, 1, 1],
        }
    )

    with caplog.at_level("WARNING", logger="feature_engineering"):
        make_features(
            df,
            ema_short_period=1,
            ema_long_period=1,
            rsi_period=2,
            volatility_window=2,
            atr_window=2,
        )

    assert any(
        "Overwriting existing target column" in record.getMessage()
        for record in caplog.records
    )


def test_make_features_redis_cache(monkeypatch):
    r = fakeredis.FakeRedis()
    df = pd.DataFrame({
        "ts": pd.date_range("2021-01-01", periods=3, freq="D"),
        "price": [1.0, 1.1, 1.2],
        "high": [1.0, 1.1, 1.2],
        "low": [0.9, 1.0, 1.1],
        "volume": [1, 1, 1],
    })
    key = "feat_key"
    out1 = make_features(
        df,
        ema_short_period=1,
        ema_long_period=1,
        rsi_period=2,
        volatility_window=2,
        atr_window=2,
        redis_client=r,
        cache_key=key,
    )
    def fail_compute(*a, **k):
        raise AssertionError("recompute")
    monkeypatch.setattr(feature_engineering, "_compute_features_pandas", fail_compute)
    out2 = make_features(
        df,
        ema_short_period=1,
        ema_long_period=1,
        rsi_period=2,
        volatility_window=2,
        atr_window=2,
        redis_client=r,
        cache_key=key,
    )
    pd.testing.assert_frame_equal(out1, out2)


def test_make_features_cache_bypass(monkeypatch):
    r = fakeredis.FakeRedis()
    df = pd.DataFrame({
        "ts": pd.date_range("2021-01-01", periods=3, freq="D"),
        "price": [1.0, 1.1, 1.2],
        "high": [1.0, 1.1, 1.2],
        "low": [0.9, 1.0, 1.1],
        "volume": [1, 1, 1],
    })
    make_features(
        df,
        ema_short_period=1,
        ema_long_period=1,
        rsi_period=2,
        volatility_window=2,
        atr_window=2,
        redis_client=r,
        cache_key="feat_key2",
    )
    called = {}
    orig = feature_engineering._compute_features_pandas
    def spy(*a, **k):
        called["run"] = True
        return orig(*a, **k)
    monkeypatch.setattr(feature_engineering, "_compute_features_pandas", spy)
    make_features(df, ema_short_period=1, ema_long_period=1, rsi_period=2, volatility_window=2, atr_window=2)
    assert called.get("run")


def test_make_features_logs_cpu_fallback(monkeypatch, caplog):
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(feature_engineering.platform, "system", lambda: "Linux")

    df = pd.DataFrame(
        {
            "ts": pd.date_range("2023-01-01", periods=3, freq="D"),
            "price": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [0.9, 1.0, 1.1],
            "volume": [1, 1, 1],
        }
    )

    with caplog.at_level("INFO", logger="feature_engineering"):
        make_features(df, use_gpu=True)

    assert any(
        "ROCm not detected; using CPU for features." in r.getMessage()
        for r in caplog.records
    )
