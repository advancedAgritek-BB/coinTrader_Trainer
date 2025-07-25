import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import data_import


def test_download_historical_data_local(tmp_path):
    df_in = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    csv_path = tmp_path / 'data.csv'
    df_in.to_csv(csv_path, index=False)

    df = data_import.download_historical_data(str(csv_path))
    pd.testing.assert_frame_equal(df, df_in)
