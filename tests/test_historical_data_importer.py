import os
import sys
import pandas as pd
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
    inserted = []

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
