import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def test_runtime_import():
    from crypto_bot.regime.api import Prediction, predict

    assert Prediction.__name__ == "Prediction"
    assert callable(predict)
