import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from crypto_bot.regime import api


class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.1, 0.2, 0.7]])


def test_label_mapping(monkeypatch):
    monkeypatch.setattr(api._registry, "load_pointer", lambda prefix: {"label_order": [1, 0, -1]})
    monkeypatch.setattr(api._registry, "load_latest", lambda prefix, allow_fallback=False: b"model")
    monkeypatch.setattr(api, "_load_model_from_bytes", lambda blob: DummyModel())

    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = api.predict(df)

    assert result.action == "short"
