import io
import os
import sys
import types
from datetime import datetime

import fakeredis
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

sys.modules.setdefault(
    "supabase",
    types.SimpleNamespace(Client=object, create_client=lambda *a, **k: object()),
)

import data_loader


def test_fetch_trade_logs_symbol_filter(monkeypatch):
    called = {}

    def fake_fetch(client, start_ts, end_ts, *, symbol=None, table="ohlc_data"):
        called["symbol"] = symbol
        called["table"] = table
        return [
            {"timestamp": start_ts.isoformat(), "symbol": symbol, "price": 1},
        ]

    monkeypatch.setattr(data_loader, "_get_client", lambda: object())
    monkeypatch.setattr(data_loader, "_fetch_logs", fake_fetch)

    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)

    df = data_loader.fetch_trade_logs(start, end, symbol="BTC")

    assert called["symbol"] == "BTC"
    assert not df.empty
    assert df["symbol"].iloc[0] == "BTC"


def test_fetch_trade_logs_uses_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.parquet"
    df_cached = pd.DataFrame({"a": [1, 2], "symbol": ["BTC", "BTC"]})
    df_cached.to_parquet(cache_path)

    def fail_get_client():
        raise AssertionError("client should not be called")

    monkeypatch.setattr(data_loader, "_get_client", fail_get_client)

    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)

    df = data_loader.fetch_trade_logs(start, end, cache_path=str(cache_path))

    pd.testing.assert_frame_equal(df, df_cached)


def test_fetch_trade_logs_redis_cache(monkeypatch):
    r = fakeredis.FakeRedis()
    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)
    key = f"ohlc_data:{start.isoformat()}:{end.isoformat()}:BTC"
    df_cached = pd.DataFrame({"a": [1, 2]})
    r.set(key, df_cached.to_json(orient="split"))

    monkeypatch.setattr(data_loader, "_get_client", lambda: None)

    def fail_fetch(*a, **k):
        raise AssertionError("should not fetch")

    monkeypatch.setattr(data_loader, "_fetch_logs", fail_fetch)

    df = data_loader.fetch_trade_logs(
        start, end, symbol="BTC", redis_client=r, redis_key=key
    )

    pd.testing.assert_frame_equal(df, df_cached)


def test_fetch_trade_logs_uses_redis(monkeypatch):
    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)
    key = (
        f"trades_{int(start.replace(tzinfo=data_loader.timezone.utc).timestamp())}_"
        f"{int(end.replace(tzinfo=data_loader.timezone.utc).timestamp())}_BTC"
    )

    fake_r = fakeredis.FakeRedis()
    df_cached = pd.DataFrame({"a": [1], "symbol": ["BTC"]})
    buf = io.BytesIO()
    df_cached.to_parquet(buf)
    fake_r.set(key, buf.getvalue())

    monkeypatch.setattr(data_loader, "_get_redis_client", lambda: fake_r)
    monkeypatch.setattr(
        data_loader,
        "_get_client",
        lambda: (_ for _ in ()).throw(AssertionError("client should not be called")),
    )

    df = data_loader.fetch_trade_logs(start, end, symbol="BTC")

    pd.testing.assert_frame_equal(df, df_cached)


def test_fetch_trade_logs_sets_redis(monkeypatch):
    fake_r = fakeredis.FakeRedis()
    monkeypatch.setattr(data_loader, "_get_redis_client", lambda: fake_r)

    def fake_fetch(client, start_ts, end_ts, *, symbol=None, table="ohlc_data"):
        return [
            {"timestamp": start_ts.isoformat(), "symbol": symbol, "price": 1},
        ]

    monkeypatch.setattr(data_loader, "_get_client", lambda: object())
    monkeypatch.setattr(data_loader, "_fetch_logs", fake_fetch)

    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)

    df = data_loader.fetch_trade_logs(start, end, symbol="BTC")

    key = (
        f"trades_{int(start.replace(tzinfo=data_loader.timezone.utc).timestamp())}_"
        f"{int(end.replace(tzinfo=data_loader.timezone.utc).timestamp())}_BTC"
    )
    cached = fake_r.get(key)
    assert cached is not None
    pd.testing.assert_frame_equal(df, pd.read_parquet(io.BytesIO(cached)))


def test_fetch_trade_logs_features_cache_hit(monkeypatch):
    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)
    key = (
        f"trades_{int(start.replace(tzinfo=data_loader.timezone.utc).timestamp())}_"
        f"{int(end.replace(tzinfo=data_loader.timezone.utc).timestamp())}_BTC"
    )
    feat_key = f"features_{key}"

    fake_r = fakeredis.FakeRedis()
    df_feat = pd.DataFrame({"feat": [1]})
    buf = io.BytesIO()
    df_feat.to_parquet(buf)
    fake_r.set(feat_key, buf.getvalue())

    monkeypatch.setattr(data_loader, "_get_redis_client", lambda: fake_r)
    monkeypatch.setattr(data_loader, "_get_client", lambda: None)
    monkeypatch.setattr(
        data_loader,
        "_fetch_logs",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fetch")),
    )

    df = data_loader.fetch_trade_logs(start, end, symbol="BTC", cache_features=True)

    pd.testing.assert_frame_equal(df, df_feat)


def test_fetch_trade_logs_caches_features(monkeypatch):
    fake_r = fakeredis.FakeRedis()
    monkeypatch.setattr(data_loader, "_get_redis_client", lambda: fake_r)

    def fake_fetch(client, start_ts, end_ts, *, symbol=None, table="ohlc_data"):
        return [
            {"timestamp": start_ts.isoformat(), "symbol": symbol, "price": 1},
        ]

    monkeypatch.setattr(data_loader, "_get_client", lambda: object())
    monkeypatch.setattr(data_loader, "_fetch_logs", fake_fetch)
    monkeypatch.setattr(data_loader, "make_features", lambda df, **k: df.assign(f=1))

    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)

    df = data_loader.fetch_trade_logs(start, end, symbol="BTC", cache_features=True)

    key = (
        f"trades_{int(start.replace(tzinfo=data_loader.timezone.utc).timestamp())}_"
        f"{int(end.replace(tzinfo=data_loader.timezone.utc).timestamp())}_BTC"
    )
    feat_key = f"features_{key}"
    cached = fake_r.get(feat_key)

    assert cached is not None and fake_r.ttl(feat_key) > 0
    pd.testing.assert_frame_equal(df, pd.read_parquet(io.BytesIO(cached)))
