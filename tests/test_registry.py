import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cointrainer import registry


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
        def upsert(self, row, **kwargs):
            captured["row"] = row
            captured["conflict"] = kwargs.get("on_conflict")
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
        def upsert(self, row, **kwargs):
            captured["row"] = row
            captured["conflict"] = kwargs.get("on_conflict")
            return types.SimpleNamespace(execute=lambda: types.SimpleNamespace(data=[{**row, "id": 1}]))

    monkeypatch.setattr(reg.client, "table", lambda name: Table())

    reg.upload_dict({}, "p", approved=True)

    row = captured["row"]
    assert row["approved"] is True


def test_upsert_called_and_deduplicates(monkeypatch, registry_with_dummy):
    reg, _ = registry_with_dummy
    table_calls = {"count": 0}

    class Table:
        def __init__(self):
            self.rows = []

        def upsert(self, row, **kwargs):
            table_calls["count"] += 1
            table_calls["conflict"] = kwargs.get("on_conflict")
            key = kwargs.get("on_conflict")
            if key is not None:
                for existing in self.rows:
                    if existing.get(key) == row.get(key):
                        existing.update(row)
                        return types.SimpleNamespace(execute=lambda: types.SimpleNamespace(data=[{**existing, "id": 1}]))
            self.rows.append(row)
            return types.SimpleNamespace(execute=lambda: types.SimpleNamespace(data=[{**row, "id": len(self.rows)}]))

    table = Table()
    monkeypatch.setattr(reg.client, "table", lambda name: table)

    reg.upload(object(), "dup", conflict_key="name")
    reg.upload(object(), "dup", conflict_key="name")

    assert table_calls["count"] == 2
    assert table_calls["conflict"] == "name"
    assert len(table.rows) == 1


def test_upload_temp_file_removed_on_error(monkeypatch, registry_with_dummy):
    reg, dummy = registry_with_dummy

    def fake_dump(obj, filename):
        with open(filename, "wb") as f:
            f.write(b"d")

    monkeypatch.setattr(
        registry,
        "joblib",
        types.SimpleNamespace(dump=fake_dump, load=lambda f: None),
        raising=False,
    )

    def fail_upload(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(dummy.storage.bucket, "upload", fail_upload)

    removed = []
    orig_unlink = os.unlink

    def spy_unlink(path):
        removed.append(path)
        orig_unlink(path)

    monkeypatch.setattr(os, "unlink", spy_unlink)

    with pytest.raises(RuntimeError):
        reg.upload(object(), "tempfail")

    assert removed, "temporary file was not removed"
    assert not os.path.exists(removed[0])


def test_upload_dict_temp_file_removed_on_error(monkeypatch, registry_with_dummy):
    reg, dummy = registry_with_dummy

    def fail_upload(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(dummy.storage.bucket, "upload", fail_upload)

    removed = []
    orig_unlink = os.unlink

    def spy_unlink(path):
        removed.append(path)
        orig_unlink(path)

    monkeypatch.setattr(os, "unlink", spy_unlink)

    with pytest.raises(RuntimeError):
        reg.upload_dict({"a": 1}, "tempdict")

    assert removed, "temporary file was not removed"
    assert not os.path.exists(removed[0])
