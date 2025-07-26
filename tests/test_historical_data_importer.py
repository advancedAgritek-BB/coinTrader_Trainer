import os
import sys
import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import historical_data_importer as hdi
import ml_trainer


def test_download_historical_data(tmp_path):
    ts = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "unix": (ts.view("int64") // 10**6),
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
    assert df["timestamp"].is_monotonic_increasing
    assert not df.duplicated("timestamp").any()
    assert "target" not in df.columns


@pytest.mark.parametrize("time_col", ["unix", "date", "UNIX", "Date", "Unix"])
def test_download_historical_data_alt_timestamp(tmp_path, time_col):
    rng = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    if time_col.lower() == "unix":
        col_vals = rng.view("int64") // 10**6
    else:
        col_vals = rng
    data = pd.DataFrame({
        time_col: col_vals,
        "close": [1, 2, 3],
    })
    csv_path = tmp_path / "prices.csv"
    data.to_csv(csv_path, index=False)

    df = hdi.download_historical_data(str(csv_path))

    assert "timestamp" in df.columns
    assert "close" in df.columns


def test_download_historical_data_skip_banner(tmp_path):
    content = (
        "https://www.CryptoDataDownload.com\n"
        "unix,close\n"
        "1000,1\n"
    )
    csv_path = tmp_path / "banner.csv"
    csv_path.write_text(content)

    df = hdi.download_historical_data(str(csv_path))

    assert list(df.columns[:2]) == ["unix", "close"]


def test_download_historical_data_drop_duplicate_ts(tmp_path):
    data = pd.DataFrame(
        {
            "unix": [1_000, 2_000, 3_000],
            "date": pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC"),
            "close": [1, 2, 3],
        }
    )
    csv_path = tmp_path / "dup.csv"
    data.to_csv(csv_path, index=False)

    df = hdi.download_historical_data(str(csv_path))

    assert "timestamp" in df.columns
    assert not df.columns.duplicated().any()


def test_download_historical_data_symbol_column_case_insensitive(tmp_path):
    rng = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "Unix": (rng.view("int64") // 10**6),
            "Symbol": ["BTC", "ETH", "BTC"],
            "Close": [1, 2, 3],
        }
    )
    csv_path = tmp_path / "symbol.csv"
    data.to_csv(csv_path, index=False)

    df = hdi.download_historical_data(str(csv_path), symbol="ETH")

    assert "Symbol" not in df.columns
    assert "symbol" in df.columns
    assert len(df) == 1
    assert df["symbol"].iloc[0] == "ETH"


def test_download_historical_data_renames_volume_column(tmp_path):
    data = pd.DataFrame(
        {
            "Unix": [1000, 2000],
            "Close": [1, 2],
            "Volume USDT": [10, 20],
        }
    )
    csv_path = tmp_path / "vol.csv"
    data.to_csv(csv_path, index=False)

    df = hdi.download_historical_data(str(csv_path))

    assert "Volume USDT" not in df.columns
    assert "volume_usdt" in df.columns
    assert list(df["volume_usdt"]) == [10, 20]


def test_download_historical_data_handles_multiple_volume_columns(tmp_path):
    data = pd.DataFrame(
        {
            "Unix": [1000, 2000],
            "Close": [1, 2],
            "Volume XRP": [5, 6],
            "Volume USDT": [10, 20],
        }
    )
    csv_path = tmp_path / "vol_multi.csv"
    data.to_csv(csv_path, index=False)

    df = hdi.download_historical_data(str(csv_path))

    assert "volume_xrp" in df.columns
    assert "volume_usdt" in df.columns


def test_insert_to_supabase_batches(monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC"),
            "close": [1, 2, 3],
            "extra": [10, 11, 12],
        }
    )
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
    hdi._INSERTED_TABLES.clear()

    def fake_create(url, key):
        return fake_client

    monkeypatch.setattr(hdi, "create_client", fake_create)

    hdi.insert_to_supabase(df, "http://localhost", "key", symbol="BTC", batch_size=2)
    hdi.insert_to_supabase(
        df,
        url='http://localhost',
        key='key',
        symbol='BTC',
        batch_size=2,
    )

    assert len(inserted) == 4
    assert sum(len(b) for b in inserted) == 6
    assert len(fake_client.rpcs) == 1
    assert "historical_prices_btc" in fake_client.rpcs[0][1]["query"]
    assert all(t == "historical_prices_btc" for t in fake_client.tables)


def test_insert_to_supabase_custom_table(monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC"),
            "close": [1, 2, 3],
            "extra": [10, 11, 12],
        }
    )
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

    hdi._INSERTED_TABLES.clear()
    def fake_create(url, key):
        return fake_client

    monkeypatch.setattr(hdi, "create_client", fake_create)

    hdi.insert_to_supabase(df, "http://localhost", "key", symbol="BTC", table="prices", batch_size=2)
    hdi.insert_to_supabase(df, url="http://localhost", key="key", symbol="BTC", table="prices", batch_size=2)

    assert len(inserted) == 4
    assert sum(len(b) for b in inserted) == 6
    assert len(fake_client.rpcs) == 1
    assert "historical_prices_btc" in fake_client.rpcs[0][1]["query"]
    assert all(t == "prices" for t in fake_client.tables)


def test_insert_to_supabase_datetime_conversion(monkeypatch):
    df = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=2, freq="h", tz="UTC")})
    captured: list[dict] = []

    class FakeTable:
        def insert(self, rows):
            captured.extend(rows)
            return types.SimpleNamespace(execute=lambda: None)

    class FakeClient:
        def __init__(self):
            self.rpcs = []

        def table(self, name):
            return FakeTable()

        def rpc(self, name, params):
            self.rpcs.append((name, params))
            return types.SimpleNamespace(execute=lambda: None)

    def fake_create(url, key):
        return FakeClient()

    monkeypatch.setattr(hdi, "create_client", fake_create)

    hdi.insert_to_supabase(df, "http://localhost", "key", symbol="BTC", batch_size=1)

    assert all("timestamp" not in row for row in captured)


def test_cli_import_data(monkeypatch):
    captured = {}

    def fake_download(
        url,
        *,
        output_file=None,
        symbol=None,
        start_ts=None,
        end_ts=None,
    ):
        captured["args"] = (url, output_file, symbol, start_ts, end_ts)
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
        "out.parquet",
        "BTC",
        "2021-01-01",
        "2021-01-02",
    )
    assert captured["batch"] == 2


def test_cli_import_csv(monkeypatch):
    captured = {}

    def fake_download(
        path, *, symbol=None, start_ts=None, end_ts=None, output_path=None
    ):
        captured["symbol"] = symbol
        return pd.DataFrame()

    def fake_insert(
        df,
        *,
        url=None,
        key=None,
        table=None,
        symbol=None,
        batch_size=500,
        client=None,
    ):
        captured['table'] = table
        captured['insert_symbol'] = symbol

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


def test_ensure_table_exists_requires_base_table():
    class FakeRpc:
        def execute(self):
            from postgrest.exceptions import APIError

            raise APIError({"code": "42P01"})

    class FakeClient:
        def rpc(self, name, params):
            return FakeRpc()

    with pytest.raises(RuntimeError):
        hdi.ensure_table_exists("BTC", client=FakeClient())
