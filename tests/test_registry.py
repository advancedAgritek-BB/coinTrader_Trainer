import os
import sys
import types
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import registry


def test_upload_uses_joblib(monkeypatch, registry_with_dummy):
    reg, dummy = registry_with_dummy

    calls = []

    def fake_dump(obj, filename):
        calls.append(obj)
        with open(filename, "wb") as f:
            f.write(b"d")

    monkeypatch.setattr(
        registry,
        "joblib",
        types.SimpleNamespace(dump=fake_dump, load=lambda f: None),
        raising=False,
    )

    reg.upload(object(), "model")

    assert calls, "joblib.dump was not called"
    assert dummy.storage.bucket.uploads, "model bytes not uploaded"


def test_list_models(registry_with_dummy):
    reg, dummy = registry_with_dummy
    result = reg.list_models()
    assert result == dummy.query.execute().data


def test_approve_calls_update(registry_with_dummy):
    reg, dummy = registry_with_dummy
    try:
        reg.approve("1")
    except AttributeError:
        pytest.fail("approve() not implemented")
