import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import historical_data_importer
import ml_trainer


def test_download_historical_data(tmp_path):
    df = pd.DataFrame({
        'timestamp': [
            '2021-01-01T00:00:00Z',
            '2021-01-02T00:00:00Z',
            '2021-01-03T00:00:00Z',
        ],
        'close': [1.0, 2.0, 3.0],
    })
    csv = tmp_path / 'data.csv'
    df.to_csv(csv, index=False)

    result = historical_data_importer.download_historical_data(
        str(csv), '2021-01-01', '2021-01-03'
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
        'prog', 'import-data', 'f.csv', '--start-ts', '2021-01-01', '--end-ts', '2021-01-02', '--table', 'tbl'
    ])

    ml_trainer.main()

    assert captured['args'] == ('f.csv', '2021-01-01', '2021-01-02')
    assert captured['table'] == 'tbl'
