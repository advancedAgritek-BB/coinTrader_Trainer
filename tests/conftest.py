"""Shared pytest fixtures for test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_trade_logs():
    """Yield a DataFrame of synthetic BTC trades for testing.

    The DataFrame contains 5k trades uniformly spaced one minute apart.
    Prices follow a simple random walk and PnL values are sampled from a
    normal distribution.
    """
    rng = np.random.default_rng(42)
    n = 5000
    timestamps = pd.date_range("2021-01-01", periods=n, freq="1T")
    price_steps = rng.normal(scale=1.0, size=n)
    prices = 20000 + np.cumsum(price_steps)
    pnl = rng.normal(loc=0.0, scale=1.0, size=n)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "BTC",
        "price": prices,
        "pnl": pnl,
    })
    yield df
