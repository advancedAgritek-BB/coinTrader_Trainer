import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ml_trainer


def test_profile_gpu_prints_message(monkeypatch, capsys):
    def fake_train(X, y, params, use_gpu=False, profile_gpu=False):
        class FakeBooster:
            best_iteration = 1

            def predict(self, data, num_iteration=None):
                return np.zeros(len(data))

        return FakeBooster(), {}

    monkeypatch.setitem(ml_trainer.TRAINERS, "regime", (fake_train, "regime_lgbm"))
    monkeypatch.setattr(
        ml_trainer, "load_cfg", lambda p: {"regime_lgbm": {"objective": "binary"}}
    )
    monkeypatch.setattr(
        ml_trainer,
        "_make_dummy_data",
        lambda n=200: (
            pd.DataFrame(np.random.normal(size=(4, 2))),
            pd.Series([0, 1, 0, 1]),
        ),
    )

    class DummyProc:
        def terminate(self):
            pass

    def fake_monitor():
        print("rocm-smi --showuse --interval 1")
        return DummyProc()

    monkeypatch.setattr(ml_trainer, "_start_rocm_smi_monitor", fake_monitor)
    monkeypatch.setattr(sys, "argv", ["ml_trainer", "train", "regime", "--profile-gpu"])

    ml_trainer.main()
    out = capsys.readouterr().out
    assert "rgp.exe --process" in out or "AMD RGP" in out
    assert "rocm-smi" in out
