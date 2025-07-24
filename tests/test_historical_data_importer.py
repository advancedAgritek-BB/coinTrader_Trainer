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


class DummyTable:
    def __init__(self):
        self.batches = []

    def insert(self, rows):
        self.batches.append(rows)
        class Exec:
            def execute(self_inner):
                return type('Resp', (), {'data': rows})
        return Exec()


class DummyClient:
    def __init__(self):
        self.tables = {}

    def table(self, name):
        tbl = self.tables.setdefault(name, DummyTable())
        return tbl


def test_insert_to_supabase_batches():
    df = pd.DataFrame({'ts': range(5), 'price': range(5), 'target': [0]*5})
    client = DummyClient()

    historical_data_importer.insert_to_supabase(
        df, 'tbl', client=client, batch_size=2
    )

    table = client.tables['tbl']
    assert len(table.batches) == 3
    assert all(len(b) <= 2 for b in table.batches)
    assert sum(len(b) for b in table.batches) == len(df)


def test_cli_import_data(monkeypatch):
    captured = {}

    def fake_download(path, start, end):
        captured['args'] = (path, start, end)
        return pd.DataFrame({'ts': [], 'price': [], 'target': []})

    def fake_insert(df, table):
        captured['table'] = table

    monkeypatch.setattr(historical_data_importer, 'download_historical_data', fake_download)
    monkeypatch.setattr(historical_data_importer, 'insert_to_supabase', fake_insert)

    monkeypatch.setattr(sys, 'argv', [
        'prog', 'import-csv', 'f.csv', '--start-ts', '2021-01-01', '--end-ts', '2021-01-02', '--table', 'tbl'
    ])

    ml_trainer.main()

    assert captured['args'] == ('f.csv', '2021-01-01', '2021-01-02')
    assert captured['table'] == 'tbl'
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import historical_data_importer as hdi


def test_download_historical_data(tmp_path):

    data = pd.DataFrame({
        'timestamp': pd.date_range('2021-01-01', periods=5, freq='D'),
        'symbol': ['BTC', 'BTC', 'ETH', 'BTC', 'BTC'],
        'close': [1, 2, 3, 4, 5],
    })
    # Add duplicate row and shuffle
    data = pd.concat([data, data.iloc[[1]]], ignore_index=True)
    csv_path = tmp_path / 'prices.csv'
    data.sample(frac=1, random_state=1).to_csv(csv_path, index=False)

    out_path = tmp_path / 'out.csv'
    df = hdi.download_historical_data(
        str(csv_path),
        symbol='BTC',
        start_ts='2021-01-01',
        end_ts='2021-01-05',
        output_path=str(out_path),
    )

    assert out_path.exists()
    assert df['ts'].is_monotonic_increasing
    assert not df.duplicated('ts').any()
    assert 'target' in df.columns
    expected_target = (df['price'].shift(-1) > df['price']).fillna(0).astype(int)
    pd.testing.assert_series_equal(df['target'], expected_target, check_names=False)


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
    def fake_download(path, *, start_ts=None, end_ts=None, **kw):
        captured['args'] = (path, start_ts, end_ts)
        return pd.DataFrame({'ts': [], 'price': [], 'target': []})

    def fake_insert(df, url, key, *, table='ohlcv', batch_size=500):
        captured['table'] = table
        captured['url'] = url
        captured['key'] = key

    monkeypatch.setattr(hdi, 'download_historical_data', fake_download)
    monkeypatch.setattr(hdi, 'insert_to_supabase', fake_insert)

    monkeypatch.setattr(sys, 'argv', [
        'prog', 'import-data', 'f.csv', '--start-ts', '2021-01-01', '--end-ts', '2021-01-02', '--table', 'tbl'
    ])

    monkeypatch.setenv('SUPABASE_URL', 'http://localhost')
    monkeypatch.setenv('SUPABASE_SERVICE_KEY', 'key')

    ml_trainer.main()

    assert captured['args'] == ('f.csv', '2021-01-01', '2021-01-02')
    assert captured['table'] == 'tbl'
    assert captured['url'] == 'http://localhost'
    assert captured['key'] == 'key'

