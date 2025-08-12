import builtins
import importlib
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))


def test_supabase_lazy(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "supabase":
            raise ModuleNotFoundError("no supabase")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    loader = importlib.import_module("cointrainer.data.loader")
    assert loader  # importing loader should not try to import supabase

    with pytest.raises(ModuleNotFoundError):
        import cointrainer.data.supabase_client as sc
        monkeypatch.setenv("SUPABASE_URL", "url")
        monkeypatch.setenv("SUPABASE_KEY", "key")
        sc.get_client()
