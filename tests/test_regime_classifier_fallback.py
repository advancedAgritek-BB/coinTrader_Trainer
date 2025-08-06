import base64
import pickle
import types
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from crypto_bot.regime import regime_classifier as rc


def test_fallback_schedules_and_returns_model(monkeypatch):
    """Fallback path should return embedded model and schedule retrain."""

    # ensure fresh state
    rc._MODEL = None

    # stub out external dependencies
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.setattr(rc, "create_client", None, raising=False)
    dummy_model = object()
    dummy_b64 = base64.b64encode(pickle.dumps(dummy_model)).decode("utf-8")
    monkeypatch.setattr(rc, "FALLBACK_MODEL_B64", dummy_b64)

    called = {}

    def fake_schedule(model: str, interval: str) -> None:
        called["args"] = (model, interval)

    monkeypatch.setattr(rc, "_schedule_retrain", fake_schedule)
    monkeypatch.setattr(rc, "joblib", types.SimpleNamespace(load=lambda b: dummy_model))

    # execute fallback path
    result = rc.load_model()

    assert result is dummy_model
    assert called["args"] == ("regime_lgbm", "daily")

