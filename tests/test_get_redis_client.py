import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cointrainer.data import loader as data_loader


def test_get_redis_client_warns(monkeypatch, caplog):
    monkeypatch.setattr(data_loader, "redis", None)
    with caplog.at_level(logging.WARNING):
        client = data_loader._get_redis_client()
    assert client is None
    assert any("caching disabled" in r.message for r in caplog.records)
