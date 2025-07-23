"""Tests for asynchronous Supabase pagination utilities."""

import os
import sys
import pandas as pd
import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import fetch_table_async, fetch_data_range_async, fetch_data_async


@pytest.mark.asyncio
async def test_fetch_data_range_async_pagination(monkeypatch):
    """Ensure async helpers handle paginated results."""

    chunk_size = 2
    pages = [
        [{"id": 1, "val": 10}, {"id": 2, "val": 20}],
        [{"id": 3, "val": 30}, {"id": 4, "val": 40}],
        [{"id": 5, "val": 50}],
    ]

from data_loader import fetch_data_range_async, fetch_data_async


# Common mock transport used across tests
PAGES = [
    [{"id": 1, "val": 10}, {"id": 2, "val": 20}],
    [{"id": 3, "val": 30}, {"id": 4, "val": 40}],
    [{"id": 5, "val": 50}],
]
CHUNK_SIZE = 2


def make_transport():
    """Return an ``httpx.MockTransport`` serving ``PAGES``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Range"):
            start = int(request.headers["Range"].split("-", 1)[0])
            idx = start // CHUNK_SIZE
        else:
            offset = int(request.url.params.get("offset", "0"))
            idx = offset // CHUNK_SIZE
        data = PAGES[idx] if idx < len(PAGES) else []
        return httpx.Response(200, json=data)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_fetch_data_range_async(monkeypatch):
    """``fetch_data_range_async`` should concatenate all paginated chunks."""
    transport = make_transport()
    real_client = httpx.AsyncClient

    def fake_client(**kwargs):
        kwargs.setdefault("transport", transport)
        kwargs.setdefault("base_url", "https://sb.example.com")
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    monkeypatch.setenv("SUPABASE_URL", "https://sb.example.com")
    monkeypatch.setenv("SUPABASE_KEY", "test")

    async with fake_client() as client:
        df1 = await fetch_table_async("trade_logs", page_size=chunk_size, client=client)

    df2 = await fetch_data_range_async(
        "trade_logs", "start", "end", chunk_size=chunk_size
    )
    df3 = await fetch_data_async("trade_logs", "start", "end", chunk_size=chunk_size)

    expected = pd.concat([pd.DataFrame(p) for p in pages], ignore_index=True)
    pd.testing.assert_frame_equal(df1, expected)
    pd.testing.assert_frame_equal(df2, expected)
    pd.testing.assert_frame_equal(df3, expected)
    df = await fetch_data_range_async(
        "trade_logs",
        "2021-01-01",
        "2021-01-02",
        chunk_size=CHUNK_SIZE,
    )

    expected = pd.concat([pd.DataFrame(p) for p in PAGES], ignore_index=True)
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.asyncio
async def test_fetch_data_async_table(monkeypatch):
    """``fetch_data_async`` should paginate the whole table."""
    transport = make_transport()
    real_client = httpx.AsyncClient

    def fake_client(**kwargs):
        kwargs.setdefault("transport", transport)
        kwargs.setdefault("base_url", "https://sb.example.com")
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    monkeypatch.setenv("SUPABASE_URL", "https://sb.example.com")
    monkeypatch.setenv("SUPABASE_KEY", "test")

    async with fake_client() as client:
        df = await fetch_data_async("trade_logs", page_size=CHUNK_SIZE, client=client)

    expected = pd.concat([pd.DataFrame(p) for p in PAGES], ignore_index=True)
    pd.testing.assert_frame_equal(df, expected)
