import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml_trainer import build_utils


def test_build_and_upload_lightgbm_wheel(monkeypatch, tmp_path):
    dist = tmp_path / "LightGBM" / "python-package" / "dist"
    dist.mkdir(parents=True)
    wheel = dist / "lightgbm-4.5.0-test.whl"
    wheel.write_bytes(b"wheel")

    monkeypatch.chdir(tmp_path)

    called = {}
    monkeypatch.setattr(build_utils.subprocess, "check_call", lambda cmd: called.setdefault("cmd", cmd))

    class FakeRegistry:
        def __init__(self, url, key, bucket="models"):
            called["init"] = (url, key, bucket)
        def upload(self, data, name, metrics):
            called["upload"] = (data, name, metrics)
    monkeypatch.setattr(build_utils, "ModelRegistry", FakeRegistry)

    result = build_utils.build_and_upload_lightgbm_wheel("url", "key")

    assert result.resolve() == wheel.resolve()
    script_path = Path(build_utils.__file__).with_name("build_lightgbm_gpu.ps1")
    assert called["cmd"] == ["powershell.exe", str(script_path)]
    assert called["init"] == ("url", "key", "wheels")
    assert called["upload"] == (b"wheel", "lightgbm_gpu_wheel", {})

