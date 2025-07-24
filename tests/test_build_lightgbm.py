import os
import sys
import types
import subprocess
import platform

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import train_pipeline


def test_build_lightgbm_upload(monkeypatch, tmp_path):
    wheel_file = tmp_path / "LightGBM" / "python-package" / "dist" / "lgpu.whl"
    wheel_file.parent.mkdir(parents=True)
    wheel_file.write_bytes(b"data")

    monkeypatch.setattr(platform, "system", lambda: "Windows")

    fake_lgb = types.ModuleType("lightgbm")
    def fake_train(params, dataset, num_boost_round):
        raise Exception("no gpu")
    fake_lgb.Dataset = lambda *a, **k: None
    fake_lgb.train = fake_train
    monkeypatch.setitem(sys.modules, "lightgbm", fake_lgb)

    run_calls = {}
    def fake_run(cmd, check):
        run_calls['cmd'] = cmd
    monkeypatch.setattr(subprocess, "run", fake_run)

    uploads = []
    class DummyBucket:
        def upload(self, name, fileobj):
            uploads.append(name)
    class DummyClient:
        def __init__(self):
            self.storage = types.SimpleNamespace(from_=lambda b: DummyBucket())
    monkeypatch.setattr(train_pipeline, "create_client", lambda u, k: DummyClient())

    result = train_pipeline.ensure_lightgbm_gpu("url", "key", script_path=str(tmp_path/"build.ps1"))

    assert result is True
    assert run_calls['cmd'][0].lower().startswith("powershell")
    assert uploads == ["lgpu.whl"]

