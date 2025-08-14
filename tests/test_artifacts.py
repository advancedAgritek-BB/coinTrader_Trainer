import json
import os
import sys
import pickle
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cointrainer import registry

class _FakeBucket:
    def __init__(self, store: dict):
        self._store = store
    def upload(self, key, data, opts=None):
        if isinstance(data, (bytes, bytearray)):
            self._store[key] = bytes(data)
        else:
            self._store[key] = data if isinstance(data, bytes) else str(data).encode("utf-8")
        return {}
    def download(self, key):
        if key not in self._store:
            raise RuntimeError("not found")
        return self._store[key]

class _FakeStorage:
    def __init__(self, store: dict):
        self._store = store
    def from_(self, bucket):
        return _FakeBucket(self._store)

class _FakeClient:
    def __init__(self, store: dict):
        self.storage = _FakeStorage(store)

def test_save_and_load_latest(monkeypatch):
    store = {}
    monkeypatch.setattr(registry, "_get_client", lambda: _FakeClient(store))
    monkeypatch.setattr(registry, "_get_bucket", lambda: "models")

    blob = pickle.dumps({"predict_proba": None})
    key = "models/regime/BTCUSDT/20250811-153000_regime_lgbm.pkl"
    metadata = {
        "feature_list": ["f1"],
        "label_order": [-1, 0, 1],
        "horizon": "15m",
    }

    registry.save_model(key, blob, metadata)
    loaded = registry.load_latest("models/regime/BTCUSDT")
    assert loaded == blob

def test_corrupt_pointer_raises(monkeypatch):
    store = {}
    monkeypatch.setattr(registry, "_get_client", lambda: _FakeClient(store))
    monkeypatch.setattr(registry, "_get_bucket", lambda: "models")

    blob = pickle.dumps({"predict_proba": None})
    key = "models/regime/BTCUSDT/20250811-153000_regime_lgbm.pkl"
    metadata = {
        "feature_list": ["f1"],
        "label_order": [-1, 0, 1],
        "horizon": "15m",
    }

    registry.save_model(key, blob, metadata)
    pointer_path = "models/regime/BTCUSDT/LATEST.json"
    store[pointer_path] = b"not-json"
    with pytest.raises(registry.RegistryError):
        registry.load_latest("models/regime/BTCUSDT")
