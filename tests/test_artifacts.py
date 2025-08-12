import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cointrainer import registry


def _fs_backend(tmp_path):
    """Return helper functions that mimic Supabase storage using ``tmp_path``."""

    root = tmp_path

    def upload(path: str, data: bytes, _opts=None):  # type: ignore[override]
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def download(path: str):  # type: ignore[override]
        return (root / path).read_bytes()

    def move(src: str, dst: str):  # type: ignore[override]
        src_p = root / src
        dst_p = root / dst
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        src_p.replace(dst_p)

    return upload, download, move


def test_save_and_load_latest(tmp_path, monkeypatch):
    upload, download, move = _fs_backend(tmp_path)

    monkeypatch.setattr(registry, "_upload", upload, raising=False)
    monkeypatch.setattr(registry, "_download", download, raising=False)
    monkeypatch.setattr(registry, "_move", move, raising=False)

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


def test_corrupt_pointer_raises(tmp_path, monkeypatch):
    upload, download, move = _fs_backend(tmp_path)

    monkeypatch.setattr(registry, "_upload", upload, raising=False)
    monkeypatch.setattr(registry, "_download", download, raising=False)
    monkeypatch.setattr(registry, "_move", move, raising=False)

    blob = pickle.dumps({"predict_proba": None})
    key = "models/regime/BTCUSDT/20250811-153000_regime_lgbm.pkl"
    metadata = {
        "feature_list": ["f1"],
        "label_order": [-1, 0, 1],
        "horizon": "15m",
    }

    registry.save_model(key, blob, metadata)
    # Corrupt pointer
    pointer = tmp_path / "models/regime/BTCUSDT/LATEST.json"
    pointer.write_text("not-json")

    with pytest.raises(registry.RegistryError):
        registry.load_latest("models/regime/BTCUSDT")

