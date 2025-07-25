import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import registry


def test_upload_uses_joblib_and_validates_metrics(monkeypatch, registry_with_dummy):
    reg, dummy = registry_with_dummy

    calls = []

    def fake_dump(obj, file_obj):
        calls.append(obj)
        file_obj.write(b"d")

    monkeypatch.setattr(
        registry,
        "joblib",
        types.SimpleNamespace(dump=fake_dump, load=lambda f: None),
        raising=False,
    )

    reg.upload(object(), "model", {"accuracy": 0.9})

    assert calls, "joblib.dump was not called"
    assert dummy.storage.bucket.uploads, "model bytes not uploaded"

    with pytest.raises(ValueError):
        reg.upload(object(), "model", ["not", "a", "dict"])


def test_list_models_applies_tag_filter(registry_with_dummy):
    reg, dummy = registry_with_dummy
    try:
        reg.list_models(tag="foo")
    except AttributeError:
        pytest.fail("list_models() not implemented")

    assert dummy.query.tag_filtered, "tag filter not applied"


def test_get_latest_uses_joblib_load(monkeypatch, registry_with_dummy):
    reg, dummy = registry_with_dummy

    loaded_obj = object()
    calls = []

    def fake_load(file_obj):
        calls.append(True)
        return loaded_obj

    monkeypatch.setattr(
        registry,
        "joblib",
        types.SimpleNamespace(dump=lambda o, f: None, load=fake_load),
        raising=False,
    )

    result = reg.get_latest("model")
    assert calls, "joblib.load was not called"
    assert result[0] is loaded_obj
