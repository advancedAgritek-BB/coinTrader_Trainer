import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from crypto_bot.regime import api


def test_predict_fallback(monkeypatch):
    def raise_err(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(api._registry, "load_pointer", raise_err)
    monkeypatch.setattr(api._registry, "load_latest", raise_err)

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = api.predict(df)

    assert result.action in {"long", "flat", "short"}
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result, api.Prediction)
