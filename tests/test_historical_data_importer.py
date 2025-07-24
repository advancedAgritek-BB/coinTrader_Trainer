import os
import sys
import types
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import historical_data_importer as hdi
import ml_trainer


def test_download_historical_data(tmp_path):
    df = pd.DataFrame(
        {
            'timestamp': [
                '2021-01-01T00:00:00Z',
                '2021-01-02T00:00:00Z',
                '2021-01-03T00:00:00Z',
            ],
            'close': [1.0, 2.0, 3.0],
        }
    )
    csv = tmp_path / 'data.csv'
    df.to_csv(csv, index=False)

    result = hdi.download_historical_data(
        str(csv), start_ts='2021-01-01', end_ts='2021-01-03'
    )

    assert list(result.columns) == ['ts', 'price', 'target']
    assert len(result) == 2
    assert result['target'].tolist() == [1, 0]


def test_insert_to_supabase_batches(monkeypatch):
    df = pd.DataFrame({'a': [1, 2, 3]})
    inserted: list[list[dict]] = []

    class FakeTable:
        def insert(self, rows):
            inserted.append(rows)
            return types.SimpleNamespace(execute=lambda: None)

    class FakeClient:
        def table(self, name):
            return FakeTable()

    def fake_create(url, key):
        return FakeClient()

    monkeypatch.setattr(hdi, 'create_client', fake_create)

    hdi.insert_to_supabase(df, 'http://localhost', 'key', table='t', batch_size=2)

    assert len(inserted) == 2
    assert sum(len(b) for b in inserted) == 3


def test_cli_import_data(monkeypatch):
    captured = {}

    def fake_download(url, symbol, start_ts, end_ts, batch_size=1000, output_file=None):
        captured['args'] = (url, symbol, start_ts, end_ts, batch_size, output_file)
        return pd.DataFrame()

    def fake_insert(df, batch_size=1000):
        captured['batch'] = batch_size

    monkeypatch.setattr(ml_trainer, 'download_historical_data', fake_download)
    monkeypatch.setattr(ml_trainer, 'insert_to_supabase', fake_insert)

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'prog',
            'import-data',
            '--source-url',
            'http://host/data.csv',
            '--symbol',
            'BTC',
            '--start-ts',
            '2021-01-01',
            '--end-ts',
            '2021-01-02',
            '--output-file',
            'out.parquet',
            '--batch-size',
            '2',
        ],
    )

    ml_trainer.main()

    assert captured['args'] == (
        'http://host/data.csv', 'BTC', '2021-01-01', '2021-01-02', 2, 'out.parquet'
    )
    assert captured['batch'] == 2
