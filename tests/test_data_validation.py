import os
import sys
import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cointrainer.data import loader as data_loader
from cointrainer.features import build as feature_engineering

pkg = types.ModuleType("coinTrader_Trainer")
pkg.data_loader = data_loader
pkg.feature_engineering = feature_engineering
sys.modules["coinTrader_Trainer"] = pkg
sys.modules["coinTrader_Trainer.data_loader"] = data_loader
sys.modules["coinTrader_Trainer.feature_engineering"] = feature_engineering

import federated_trainer
import swarm_sim


@pytest.mark.asyncio
async def test_fetch_and_prepare_data_empty(monkeypatch):
    async def fake_fetch(table, start, end):
        return pd.DataFrame()

    monkeypatch.setattr(data_loader, "fetch_data_range_async", fake_fetch)

    with pytest.raises(ValueError):
        await swarm_sim.fetch_and_prepare_data("s", "e", min_rows=1)


@pytest.mark.asyncio
async def test_fetch_and_prepare_data_too_few(monkeypatch):
    df = pd.DataFrame({"ts": [1, 2, 3], "price": [1, 2, 3], "target": [0, 1, 0]})

    async def fake_fetch(table, start, end):
        return df

    monkeypatch.setattr(data_loader, "fetch_data_range_async", fake_fetch)
    monkeypatch.setattr(feature_engineering, "make_features", lambda d, **k: d)

    with pytest.raises(ValueError):
        await swarm_sim.fetch_and_prepare_data("s", "e", min_rows=5)


def test_prepare_data_empty(monkeypatch):
    async def fake_prepare(*args, **kwargs):
        raise ValueError("No data available")

    monkeypatch.setattr(federated_trainer, "prepare_data", fake_prepare)

    with pytest.raises(ValueError):
        federated_trainer.prepare_data("s", "e", min_rows=1)


def test_prepare_data_too_few(monkeypatch):
    async def fake_prepare(*args, **kwargs):
        raise ValueError("too few")

    monkeypatch.setattr(federated_trainer, "prepare_data", fake_prepare)

    with pytest.raises(ValueError):
        federated_trainer.prepare_data("s", "e", min_rows=3)
