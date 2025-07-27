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
    import importlib
    return importlib.import_module("kraken_fetch")


def test_insert_to_supabase_custom_table(monkeypatch):
    kf = _load_module(monkeypatch)
    df = _sample_df()

    captured = []

    class FakeTable:
        def __init__(self, name):
            self.name = name

        def upsert(self, rows, *, on_conflict=None):
            captured.append((self.name, on_conflict))
            return types.SimpleNamespace(execute=lambda: None)

    class FakeClient:
        def __init__(self):
            self.tables = []

        def table(self, name):
            self.tables.append(name)
            return FakeTable(name)

    client = FakeClient()
    kf.insert_to_supabase(client, df, table="logs", batch_size=1)
    assert all(t == "logs" for t in client.tables)
    assert all(c == ("logs", "ts,symbol") for c in captured)


def test_insert_to_supabase_conflict_cols(monkeypatch):
    kf = _load_module(monkeypatch)
    df = _sample_df().iloc[:1]

    captured = []

    class FakeTable:
        def upsert(self, rows, *, on_conflict=None):
            captured.append(on_conflict)
            return types.SimpleNamespace(execute=lambda: None)

    class FakeClient:
        def table(self, name):
            return FakeTable()

    client = FakeClient()
    kf.insert_to_supabase(
        client,
        df,
        table="logs",
        batch_size=1,
        conflict_cols=("ts", "symbol", "price"),
    )

    assert captured == ["ts,symbol,price"]


def test_append_kraken_data_passes_table(monkeypatch):
    kf = _load_module(monkeypatch)
    df = _sample_df().iloc[:1]
    captured = {}

    monkeypatch.setattr(kf, "get_tradable_pairs", lambda: ["BTCUSD"])
    monkeypatch.setattr(kf, "fetch_kraken_ohlc", lambda p, interval: df)

    def fake_get_last_ts(client, symbol, table):
        captured["last"] = table
        return None

    def fake_insert(client, df, *, table, batch_size=1000):
        captured["insert"] = table

    monkeypatch.setattr(kf, "get_last_ts", fake_get_last_ts)
    monkeypatch.setattr(kf, "insert_to_supabase", fake_insert)
    monkeypatch.setattr(kf, "time", types.SimpleNamespace(sleep=lambda s: None))

    kf.append_kraken_data(table="alt_table")

    assert captured["last"] == "alt_table"
    assert captured["insert"] == "alt_table"


def test_cli_table_override(monkeypatch):
    captured = {}

    kf = _load_module(monkeypatch)

    def fake_append(interval=1, delay_sec=1.0, *, table):
        captured["table"] = table

    monkeypatch.setattr(kf, "append_kraken_data", fake_append)
    monkeypatch.setattr(sys, "argv", ["prog", "--table", "cli_table"])

    kf.main()

    assert captured["table"] == "cli_table"


def test_cli_env_default(monkeypatch):
    captured = {}

    kf = _load_module(monkeypatch)

    def fake_append(interval=1, delay_sec=1.0, *, table):
        captured["table"] = table

    monkeypatch.setattr(kf, "append_kraken_data", fake_append)
    monkeypatch.setenv("KRAKEN_TABLE", "env_table")
    monkeypatch.setattr(sys, "argv", ["prog"])

    kf.main()

    assert captured["table"] == "env_table"


