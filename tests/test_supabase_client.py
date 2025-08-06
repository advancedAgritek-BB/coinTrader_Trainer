import base64
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import utils.supabase_client as sc


def test_upload_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_URL", "https://sb.example")
    monkeypatch.setenv("SUPABASE_KEY", "anon")

    calls = {}

    class FakeBucket:
        def upload(self, path, data):
            calls["path"] = path
            calls["data"] = data

    class FakeStorage:
        def from_(self, bucket):
            calls["bucket"] = bucket
            return FakeBucket()

    class FakeClient:
        def __init__(self):
            self.storage = FakeStorage()

    def fake_create(url, key):
        calls["url"] = url
        calls["key"] = key
        return FakeClient()

    monkeypatch.setattr(sc, "create_client", fake_create)

    model_bytes = b"binary-blob"
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(model_bytes)

    sc.upload_model("models", "model.bin", str(model_path))

    assert calls["url"] == "https://sb.example"
    assert calls["key"] == "anon"
    assert calls["bucket"] == "models"
    assert calls["path"] == "model.bin"
    assert calls["data"] == model_bytes


def test_download_model(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://sb.example")
    monkeypatch.setenv("SUPABASE_KEY", "anon")

    calls = {}
    expected = b"model-bytes"

    class FakeBucket:
        def download(self, path):
            calls["path"] = path
            return expected

    class FakeStorage:
        def from_(self, bucket):
            calls["bucket"] = bucket
            return FakeBucket()

    class FakeClient:
        def __init__(self):
            self.storage = FakeStorage()

    def fake_create(url, key):
        calls["url"] = url
        calls["key"] = key
        return FakeClient()

    monkeypatch.setattr(sc, "create_client", fake_create)

    data = sc.download_model("models", "model.bin")

    assert data == expected
    assert calls["url"] == "https://sb.example"
    assert calls["key"] == "anon"
    assert calls["bucket"] == "models"
    assert calls["path"] == "model.bin"


def test_load_fallback_model_roundtrip():
    obj = {"a": 1}
    b64 = base64.b64encode(pickle.dumps(obj)).decode("utf-8")
    assert sc.load_fallback_model(b64) == obj
