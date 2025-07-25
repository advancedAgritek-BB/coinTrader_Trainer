import os
import sys
from datetime import datetime


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import data_loader


def test_fetch_trade_aggregates(monkeypatch):
    called = {}

    class Functions:
        def invoke(self, name, opts):
            called["name"] = name
            called["body"] = opts["body"]
            return [{"symbol": "BTC", "count": 5}]

    class Client:
        def __init__(self):
            self.functions = Functions()

    monkeypatch.setattr(data_loader, "_get_client", lambda: Client())

    start = datetime(2021, 1, 1)
    end = datetime(2021, 1, 2)

    df = data_loader.fetch_trade_aggregates(start, end, symbol="BTC")

    assert called["name"] == "aggregate-trades"
    assert called["body"]["symbol"] == "BTC"
    assert not df.empty
    assert df["count"].iloc[0] == 5
