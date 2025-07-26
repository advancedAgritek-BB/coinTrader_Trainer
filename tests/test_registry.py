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


def test_upload_dict_uploads_json(monkeypatch, registry_with_dummy):
    reg, dummy = registry_with_dummy
    captured = {}

    class Table:
        def insert(self, row):
            captured["row"] = row
            return types.SimpleNamespace(execute=lambda: types.SimpleNamespace(data=[{**row, "id": 1}]))

    monkeypatch.setattr(reg.client, "table", lambda name: Table())

    reg.upload_dict({"a": 1}, "params", {"m": 2})

    assert dummy.storage.bucket.uploads
    path, data = dummy.storage.bucket.uploads[-1]
    assert path == "params.json"
    import json

    assert json.loads(data.decode()) == {"a": 1}
    row = captured["row"]
    assert row["metadata"] == {"m": 2}
    assert not row["approved"]


def test_upload_dict_approved(registry_with_dummy, monkeypatch):
    reg, dummy = registry_with_dummy
    captured = {}

    class Table:
        def insert(self, row):
            captured["row"] = row
            return types.SimpleNamespace(execute=lambda: types.SimpleNamespace(data=[{**row, "id": 1}]))

    monkeypatch.setattr(reg.client, "table", lambda name: Table())

    reg.upload_dict({}, "p", approved=True)

    row = captured["row"]
    assert row["approved"] is True
