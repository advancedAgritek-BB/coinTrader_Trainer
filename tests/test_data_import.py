import pandas as pd

import cointrainer.data.importers as data_import


def test_download_historical_data_local(tmp_path):
    df_in = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "close": [1, 2],
        }
    )
    csv_path = tmp_path / "data.csv"
    df_in.to_csv(csv_path, index=False)

    df = data_import.download_historical_data(str(csv_path))

    assert "ts" in df.columns
    assert "price" in df.columns
    pd.testing.assert_series_equal(
        df["ts"], pd.to_datetime(df_in["timestamp"], utc=True), check_names=False
    )
    pd.testing.assert_series_equal(df["price"], df_in["close"], check_names=False)


def test_download_historical_data_skip_banner(tmp_path):
    content = (
        "This data was provided by CryptoDataDownload.com\n"
        "unix,close\n"
        "1,100\n"
    )
    csv_path = tmp_path / "Binance_XRPUSDT_d.csv"
    csv_path.write_text(content)

    df = data_import.download_historical_data(str(csv_path))

    assert "ts" in df.columns
    assert "price" in df.columns
    assert len(df) == 1
    assert df["price"].iloc[0] == 100
