import os
import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml_trainer import build_utils


def _load_ml_trainer():
    path = Path(__file__).resolve().parents[1] / "ml_trainer.py"
    spec = importlib.util.spec_from_file_location("ml_trainer_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_flag_triggers_wheel_build(monkeypatch):
    called = {}

    def fake_build(url, key):
        called["args"] = (url, key)
        return Path("dummy.whl")

    monkeypatch.setattr(build_utils, "build_and_upload_lightgbm_wheel", fake_build)
    monkeypatch.setenv("SUPABASE_URL", "url")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key")

    yaml_mod = type("Y", (), {"safe_load": lambda *a, **k: {}})()
    monkeypatch.setitem(sys.modules, "yaml", yaml_mod)

    ml = _load_ml_trainer()

    def fake_trainer(X, y, params, use_gpu=False):
        called["trained"] = use_gpu
        return None, {}

    ml.TRAINERS["regime"] = (fake_trainer, "regime_lgbm")

    monkeypatch.setattr(sys, "argv", ["ml_trainer.py", "train", "regime", "--use-gpu"])

    ml.main()

    assert called["args"] == ("url", "key")
    assert called["trained"] is True

