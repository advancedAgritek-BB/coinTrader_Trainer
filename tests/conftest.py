"""Shared pytest fixtures for test suite."""

import os
import types

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
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTC",
            "price": prices,
            "pnl": pnl,
        }
    )
    yield df


class DummyQuery:
    def __init__(self):
        self.tag_filtered = False

    def select(self, *args, **kwargs):
        return self

    def eq(self, column, value):
        if column == "tags":
            self.tag_filtered = True
        return self

    def contains(self, column, value):
        if column == "tags":
            self.tag_filtered = True
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        return types.SimpleNamespace(
            data=[
                {
                    "id": 1,
                    "name": "m",
                    "file_path": "m/abc.pkl",
                    "sha256": "abc",
                    "metrics": {},
                    "approved": True,
                }
            ]
        )


class DummyStorageBucket:
    def __init__(self):
        self.uploads = []
        self.download_data = b"model-bytes"

    def upload(self, path, file_obj):
        self.uploads.append((path, file_obj.read()))

    def download(self, path):
        return self.download_data


class DummyStorage:
    def __init__(self):
        self.bucket = DummyStorageBucket()

    def from_(self, bucket):
        return self.bucket


class DummySupabase:
    def __init__(self):
        self.storage = DummyStorage()
        self.query = DummyQuery()

    def table(self, name):
        class Table:
            def insert(_, row):
                return types.SimpleNamespace(
                    execute=lambda: types.SimpleNamespace(data=[{**row, "id": 1}])
                )

            def select(_, *args, **kwargs):
                return self.query

            def update(_, *args, **kwargs):
                class Q:
                    def eq(self, *a, **k):
                        return types.SimpleNamespace(execute=lambda: None)

                return Q()

        return Table()


@pytest.fixture
def registry_with_dummy(monkeypatch):
    import registry

    dummy = DummySupabase()
    monkeypatch.setattr(registry, "create_client", lambda url, key: dummy)
    return registry.ModelRegistry("http://localhost", "anon"), dummy


def pytest_addoption(parser):
    parser.addoption("--supabase-user-email", action="store", default=None)
    parser.addoption("--supabase-password", action="store", default=None)
    parser.addoption("--supabase-jwt", action="store", default=None)


@pytest.fixture(autouse=True)
def supabase_env(monkeypatch, pytestconfig):
    email = pytestconfig.getoption("--supabase-user-email") or os.environ.get(
        "SUPABASE_USER_EMAIL"
    )
    password = pytestconfig.getoption("--supabase-password") or os.environ.get(
        "SUPABASE_PASSWORD"
    )
    jwt = pytestconfig.getoption("--supabase-jwt") or os.environ.get("SUPABASE_JWT")
    if email:
        monkeypatch.setenv("SUPABASE_USER_EMAIL", email)
    if password:
        monkeypatch.setenv("SUPABASE_PASSWORD", password)
    if jwt:
        monkeypatch.setenv("SUPABASE_JWT", jwt)
