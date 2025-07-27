import asyncio
import os
import sys
import types
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Stub heavy optional dependency used during import
class DummyBroker:
    def getvalue(self):
        return 0

    def setcommission(self, commission):
        pass

    def set_slippage_perc(self, slippage):
        pass


class DummyCerebro:
    def __init__(self):
        self.broker = DummyBroker()

    def adddata(self, data):
        pass

    def addstrategy(self, strategy, **kwargs):
        pass

    def run(self):
        pass


class DummyFeeds:
    class PandasData:
        def __init__(self, dataname=None):
            pass


sys.modules.setdefault(
    "backtrader",
    types.SimpleNamespace(
        Strategy=object, Cerebro=DummyCerebro, feeds=DummyFeeds
    ),
)
sys.modules.setdefault("optuna", types.SimpleNamespace())

import swarm_sim
import utils


async def fake_fetch(
    start,
    end,
    *,
    table="ohlc_data",
    return_threshold=0.01,
    min_rows=1,
    balance=False,
    **_,
):
    return pd.DataFrame({"f": [1, 2, 3]}), pd.Series([0, 1, 0])


async def fake_simulate(self, X, y, base_params):
    self.fitness = float(self.id)


def test_run_swarm_search_returns_params(monkeypatch):
    monkeypatch.setattr(utils, "prepare_data", fake_fetch)
    monkeypatch.setattr(swarm_sim.SwarmAgent, "simulate", fake_simulate)
    monkeypatch.setattr(swarm_sim.yaml, "safe_load", lambda fh: {})
    monkeypatch.delenv("SUPABASE_URL", raising=False)

    params = asyncio.run(
        swarm_sim.run_swarm_search(
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-02"),
            num_agents=2,
        )
    )

    assert isinstance(params, dict)


def test_run_swarm_search_requires_networkx(monkeypatch):
    monkeypatch.setattr(swarm_sim, "nx", None)
    with pytest.raises(SystemExit):
        asyncio.run(
            swarm_sim.run_swarm_search(
                pd.Timestamp("2021-01-01"),
                pd.Timestamp("2021-01-02"),
            )
        )

