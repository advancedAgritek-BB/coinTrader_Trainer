import pandas as pd

from cointrainer.features.build import build_features


def test_feature_list_matches_columns():
    df = pd.DataFrame(
        {
            "close": [1, 2, 3, 4, 5, 6],
            "high": [1, 2, 3, 4, 5, 6],
            "low": [0.5, 1, 2, 3, 4, 5],
            "volume": [1, 1, 1, 1, 1, 1],
        }
    )
    use = {"rsi": True, "ema": True, "obv": True}
    params = {"rsi": 2, "ema": 3}
    X, meta = build_features(df, use=use, params=params)
    assert meta["feature_list"] == list(X.columns)
