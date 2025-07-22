import os
import sys
import pandas as pd
import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import fetch_data_range_async


@pytest.mark.asyncio
async def test_fetch_data_async_pagination(monkeypatch):
    chunk_size = 2
    pages = [
        [{"id": 1, "val": 10}, {"id": 2, "val": 20}],
        [{"id": 3, "val": 30}, {"id": 4, "val": 40}],
        [{"id": 5, "val": 50}],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", "0"))
        idx = offset // chunk_size
        data = pages[idx] if idx < len(pages) else []
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)

    real_client = httpx.AsyncClient

    def fake_client(**kwargs):
        return real_client(transport=transport, base_url="https://sb.example.com")

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    monkeypatch.setenv("SUPABASE_URL", "https://sb.example.com")
    monkeypatch.setenv("SUPABASE_KEY", "test")

    df = await fetch_data_range_async(
        "trade_logs", "start", "end", chunk_size=chunk_size
    )

    expected = pd.concat([pd.DataFrame(p) for p in pages], ignore_index=True)
    pd.testing.assert_frame_equal(df, expected)
