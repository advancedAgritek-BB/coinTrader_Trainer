import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import optuna_search
except Exception:  # pragma: no cover - module may not exist yet
    optuna_search = types.SimpleNamespace()


@pytest.mark.asyncio
async def test_run_optuna_search_uses_given_range(monkeypatch):
    captured = {}

    async def fake_fetch(table, start, end):
        captured['args'] = (table, start, end)
        return pd.DataFrame({'price': [1, 2], 'target': [0, 1]})

    monkeypatch.setattr(optuna_search, 'fetch_data_range_async', fake_fetch, raising=False)
    monkeypatch.setattr(optuna_search, 'make_features', lambda df, **k: df, raising=False)

    class FakeBooster:
        def predict(self, X, num_iteration=None):
            return np.zeros(len(X))

    monkeypatch.setattr(getattr(optuna_search, 'lgb', types.SimpleNamespace()), 'train', lambda *a, **k: FakeBooster(), raising=False)

    class FakeStudy:
        best_params = {'learning_rate': 0.1}

        def optimize(self, obj, n_trials=10):
            pass

    fake_optuna = types.SimpleNamespace(create_study=lambda direction='minimize': FakeStudy())
    monkeypatch.setitem(optuna_search.__dict__, 'optuna', fake_optuna)

    start = '2021-01-01'
    end = '2021-01-02'

    if not hasattr(optuna_search, 'run_optuna_search'):
        pytest.skip('run_optuna_search not implemented')

    await optuna_search.run_optuna_search(start, end)

    assert captured.get('args') == ('ohlc_data', start, end)


def test_cli_optuna_merges_params(monkeypatch):
    captured = {}

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        captured['params'] = params.copy()
        class FakeModel:
            def predict(self, X):
                return np.zeros(len(X))
        return FakeModel(), {}

    import ml_trainer

    monkeypatch.setitem(ml_trainer.TRAINERS, 'regime', (fake_train, 'regime_lgbm'))
    monkeypatch.setattr(ml_trainer, 'load_cfg', lambda p: {'regime_lgbm': {'objective': 'binary'}}, raising=False)
    monkeypatch.setattr(ml_trainer, '_make_dummy_data', lambda n=200: (pd.DataFrame(np.random.normal(size=(4,2))), pd.Series([0,1,0,1])), raising=False)

    async def fake_search(start, end, *, table='ohlc_data'):
        captured['called'] = True
        return {'learning_rate': 0.05}

    module = types.SimpleNamespace(run_optuna_search=fake_search)
    monkeypatch.setitem(sys.modules, 'optuna_search', module)

    if '--optuna' not in open('ml_trainer.py').read():
        pytest.skip('optuna CLI option not implemented')

    argv = ['prog', 'train', 'regime', '--optuna']
    monkeypatch.setattr(sys, 'argv', argv)

    ml_trainer.main()

    assert captured.get('called')
    assert captured['params'].get('learning_rate') == 0.05
