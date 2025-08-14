import json
import types
import builtins
import os
import sys
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
            # supabase-py may accept file-like; handle str/bytes only here
            self._store[key] = data if isinstance(data, bytes) else str(data).encode("utf-8")
        return {}
    def download(self, key):
        if key not in self._store:
            raise RuntimeError("not found")
        return self._store[key]
    def list(self, prefix=None):
        pref = prefix or ""
        return [{"name": k} for k in self._store.keys() if k.startswith(pref)]

class _FakeStorage:
    def __init__(self, store: dict):
        self._store = store
    def from_(self, bucket):
        # ignore bucket name, use single store
        return _FakeBucket(self._store)

class _FakeClient:
    def __init__(self, store: dict):
        self.storage = _FakeStorage(store)

def test_save_and_load_with_stubbed_supabase(monkeypatch):
    # in-memory object store
    store = {}

    # Monkeypatch: _get_client -> fake client; _get_bucket -> 'models'
    monkeypatch.setattr(registry, "_get_client", lambda: _FakeClient(store))
    monkeypatch.setattr(registry, "_get_bucket", lambda: "models")

    # Prepare blob + meta
    blob = b"model-bytes"
    key = "models/regime/XRPUSD/20250101-000000_regime_lgbm.pkl"
    meta = {"feature_list": ["a","b"], "label_order": [-1,0,1], "symbol": "XRPUSD"}

    # Save
    registry.save_model(key, blob, meta)

    # Pointer exists
    pointer_path = "models/regime/XRPUSD/LATEST.json"
    assert pointer_path in store
    pointer = json.loads(store[pointer_path].decode("utf-8"))
    assert pointer["key"] == key
    assert pointer["hash"].startswith("sha256:")

    # Load pointer + latest
    out_meta = registry.load_pointer("models/regime/XRPUSD")
    assert out_meta["key"] == key
    model_bytes = registry.load_latest("models/regime/XRPUSD")
    assert model_bytes == blob

def test_missing_env_raises(monkeypatch):
    # Ensure _get_client raises when vars missing
    def bad_client():
        raise registry.RegistryError("SUPABASE_URL and SUPABASE_KEY must be set")
    monkeypatch.setattr(registry, "_get_client", bad_client)
    with pytest.raises(registry.RegistryError):
        registry.load_pointer("models/regime/XRPUSD")
