import os
import sys
import types
import importlib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _sample_df():
    return pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=2, freq="h", tz="UTC"),
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "vwap": [1, 2],
            "volume": [1, 2],
            "trades": [1, 2],
            "price": [1, 2],
            "symbol": ["BTCUSD", "BTCUSD"],
        }
    )


def _load_module(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "http://sb")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key")
    sys.modules["supabase"] = types.SimpleNamespace(create_client=lambda *a, **k: object(), Client=object)
    sys.modules.setdefault("ccxt", types.SimpleNamespace())
    import importlib
    return importlib.import_module("ccxt_fetch_all")


def test_insert_to_supabase_custom_table(monkeypatch):
    mod = _load_module(monkeypatch)
    df = _sample_df()

    captured = []

    class FakeTable:
        def __init__(self, name):
            self.name = name

        def insert(self, rows):
            captured.append(self.name)
            return types.SimpleNamespace(execute=lambda: None)

    class FakeClient:
        def __init__(self):
            self.tables = []

        def table(self, name):
            self.tables.append(name)
            return FakeTable(name)

    client = FakeClient()
    mod.insert_to_supabase(client, df, table="logs", batch_size=1)
    assert all(t == "logs" for t in client.tables)


def test_append_ccxt_data_all_passes_table(monkeypatch):
    mod = _load_module(monkeypatch)
    df = _sample_df().iloc[:1]
    captured = {}

    monkeypatch.setattr(mod, "get_markets", lambda exc: ["BTC/USD"])
    monkeypatch.setattr(mod, "fetch_ccxt_ohlc", lambda exc, p, tf: df)

    def fake_get_last_ts(client, symbol, table):
        captured["last"] = table
        return None

    def fake_insert(client, df, *, table, batch_size=1000):
        captured["insert"] = table

    monkeypatch.setattr(mod, "get_last_ts", fake_get_last_ts)
    monkeypatch.setattr(mod, "insert_to_supabase", fake_insert)
    monkeypatch.setattr(mod, "time", types.SimpleNamespace(sleep=lambda s: None))
    monkeypatch.setattr(mod, "get_exchange", lambda name: object())

    mod.append_ccxt_data_all(table="alt_table")

    assert captured["last"] == "alt_table"
    assert captured["insert"] == "alt_table"


def test_cli_table_override(monkeypatch):
    captured = {}

    mod = _load_module(monkeypatch)

    def fake_append(exchange_name="binance", timeframe="1m", delay_sec=1.0, *, table):
        captured["table"] = table

    monkeypatch.setattr(mod, "append_ccxt_data_all", fake_append)
    monkeypatch.setattr(sys, "argv", ["prog", "--table", "cli_table"])

    mod.main()

    assert captured["table"] == "cli_table"


def test_cli_env_default(monkeypatch):
    captured = {}

    mod = _load_module(monkeypatch)

    def fake_append(exchange_name="binance", timeframe="1m", delay_sec=1.0, *, table):
        captured["table"] = table

    monkeypatch.setattr(mod, "append_ccxt_data_all", fake_append)
    monkeypatch.setenv("CCXT_TABLE", "env_table")
    monkeypatch.setattr(sys, "argv", ["prog"])

    mod.main()

    assert captured["table"] == "env_table"
