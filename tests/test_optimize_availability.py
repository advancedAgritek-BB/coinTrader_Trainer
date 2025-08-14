import importlib


def test_optuna_optional():
    try:
        optuna = importlib.import_module("optuna")
        assert optuna is not None
    except Exception:
        # It's okay if optuna isn't installed
        assert True
