import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from crypto_bot.regime import regime_classifier as rc


def test_fallback_triggers_retrain(monkeypatch):
    fake_model = object()

    def fake_download(bucket, name):
        raise Exception("fail")

    called = {}

    def fake_schedule(model, interval):
        called['args'] = (model, interval)

    monkeypatch.setattr(rc, 'download_model', fake_download)
    monkeypatch.setattr(rc, 'schedule_retrain', fake_schedule)
    monkeypatch.setattr(rc, 'load_fallback_model', lambda b64: fake_model)

    model = rc.get_regime_model()

    assert model is fake_model
    assert called['args'] == ('regime', 'immediate')
