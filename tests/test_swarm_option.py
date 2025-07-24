import os
import sys
import types
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ml_trainer


def test_cli_swarm_merges_params(monkeypatch):
    captured = {}

    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        captured['params'] = params.copy()
        class FakeBooster:
            best_iteration = 1
            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))
        return FakeBooster(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, 'regime', (fake_train, 'regime_lgbm'))
    monkeypatch.setattr(ml_trainer, 'load_cfg', lambda path: {'regime_lgbm': {'objective': 'binary'}})
    monkeypatch.setattr(
        ml_trainer,
        '_make_dummy_data',
        lambda n=200: (pd.DataFrame(np.random.normal(size=(4, 2))), pd.Series([0, 1, 0, 1]))
    )

    async def fake_run(start, end):
        captured['swarm_called'] = True
        return {'learning_rate': 0.1}

    module = types.SimpleNamespace(run_swarm_search=fake_run)
    monkeypatch.setitem(sys.modules, 'swarm_sim', module)

    argv = ['ml_trainer', 'train', 'regime', '--swarm']
    monkeypatch.setattr(sys, 'argv', argv)

    ml_trainer.main()

    assert captured.get('swarm_called')
    assert captured['params'].get('learning_rate') == 0.1
