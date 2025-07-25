import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import historical_data_importer as hdi
import ml_trainer


def test_download_historical_data(tmp_path):
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=5, freq="D"),
            "symbol": ["BTC", "BTC", "ETH", "BTC", "BTC"],
            "close": [1, 2, 3, 4, 5],
        }
    )
    # Add duplicate row and shuffle
    data = pd.concat([data, data.iloc[[1]]], ignore_index=True)
    csv_path = tmp_path / "prices.csv"
    data.sample(frac=1, random_state=1).to_csv(csv_path, index=False)

    out_path = tmp_path / "out.csv"
    df = hdi.download_historical_data(
        str(csv_path),
        symbol="BTC",
        start_ts="2021-01-01",
        end_ts="2021-01-05",
        output_path=str(out_path),
    )

    assert out_path.exists()
    assert df["ts"].is_monotonic_increasing
    assert not df.duplicated("ts").any()
    assert "target" in df.columns
    expected_target = (df["price"].shift(-1) > df["price"]).fillna(0).astype(int)
    pd.testing.assert_series_equal(df["target"], expected_target, check_names=False)


def test_insert_to_supabase_batches(monkeypatch):
    df = pd.DataFrame({"a": [1, 2, 3]})
    inserted: list[list[dict]] = []

    class FakeTable:
        def insert(self, rows):
            inserted.append(rows)
            return types.SimpleNamespace(execute=lambda: None)

    class FakeClient:
        def __init__(self):
            self.rpcs = []
            self.tables = []

        def table(self, name):
            self.tables.append(name)
            return FakeTable()

        def rpc(self, name, params):
            self.rpcs.append((name, params))
            return types.SimpleNamespace(execute=lambda: None)

    fake_client = FakeClient()

    def fake_create(url, key):
        return fake_client

    monkeypatch.setattr(hdi, "create_client", fake_create)

    hdi.insert_to_supabase(df, "http://localhost", "key", symbol="BTC", batch_size=2)

    assert len(inserted) == 2
    assert sum(len(b) for b in inserted) == 3
    assert fake_client.rpcs
    assert "historical_prices_btc" in fake_client.rpcs[0][1]["query"]
    assert all(t == "historical_prices_btc" for t in fake_client.tables)


def test_cli_import_data(monkeypatch):
    captured = {}

    def fake_download(url, symbol, start_ts, end_ts, batch_size=1000, output_file=None):
        captured["args"] = (url, symbol, start_ts, end_ts, batch_size, output_file)
        return pd.DataFrame()

    def fake_insert(df, batch_size=1000):
        captured["batch"] = batch_size

    monkeypatch.setattr(ml_trainer, "download_historical_data", fake_download)
    monkeypatch.setattr(ml_trainer, "insert_to_supabase", fake_insert)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "import-data",
            "--source-url",
            "http://host/data.csv",
            "--symbol",
            "BTC",
            "--start-ts",
            "2021-01-01",
            "--end-ts",
            "2021-01-02",
            "--output-file",
            "out.parquet",
            "--batch-size",
            "2",
        ],
    )

    ml_trainer.main()

    assert captured["args"] == (
        "http://host/data.csv",
        "BTC",
        "2021-01-01",
        "2021-01-02",
        2,
        "out.parquet",
    )
    assert captured["batch"] == 2


def test_cli_import_csv(monkeypatch):
    captured = {}

    def fake_download(
        path, *, symbol=None, start_ts=None, end_ts=None, output_path=None
    ):
        captured["symbol"] = symbol
        return pd.DataFrame()

    def fake_insert(df, url, key, table=None, symbol=None, batch_size=500):
        captured["table"] = table
        captured["insert_symbol"] = symbol

    monkeypatch.setattr(hdi, "download_historical_data", fake_download)
    monkeypatch.setattr(hdi, "insert_to_supabase", fake_insert)

    monkeypatch.setattr(
        sys, "argv", ["prog", "import-csv", "prices.csv", "--symbol", "ETH"]
    )
    monkeypatch.setenv("SUPABASE_URL", "http://sb")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key")

    ml_trainer.main()

    assert captured["symbol"] == "ETH"
    assert captured["table"] == "historical_prices_eth"
    assert captured["insert_symbol"] == "ETH"
