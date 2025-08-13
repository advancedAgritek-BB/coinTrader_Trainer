import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from crypto_bot.regime import api


class DummyModel:
    def __init__(self):
        self.seen = None

    def predict_proba(self, X):
        self.seen = list(X.columns)
        return np.array([[0.1, 0.2, 0.7]])


def test_feature_reorder(monkeypatch):
    model = DummyModel()
    monkeypatch.setattr(api._registry, "load_pointer", lambda prefix: {"feature_list": ["a", "b", "c"]})
    monkeypatch.setattr(api._registry, "load_latest", lambda prefix, allow_fallback=False: b"model")
    monkeypatch.setattr(api, "_load_model_from_bytes", lambda blob: model)

    df = pd.DataFrame({"c": [3], "a": [1], "b": [2]})
    api.predict(df)

    assert model.seen == ["a", "b", "c"]
