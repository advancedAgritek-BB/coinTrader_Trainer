import asyncio
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import swarm_sim


async def fake_fetch(start, end, *, table="ohlc_data", return_threshold=0.01):
    return pd.DataFrame({"f": [1, 2, 3]}), pd.Series([0, 1, 0])


def fake_simulate(self, X, y, base_params):
    self.fitness = float(self.id)


def test_run_swarm_search_returns_params(monkeypatch):
    monkeypatch.setattr(swarm_sim, "fetch_and_prepare_data", fake_fetch)
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

