import os
import sys
import pandas as pd
import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import fetch_data_async


@pytest.mark.asyncio
async def test_fetch_data_async_pagination():
    page_size = 2
    pages = [
        [{"id": 1, "val": 10}, {"id": 2, "val": 20}],
        [{"id": 3, "val": 30}, {"id": 4, "val": 40}],
        [{"id": 5, "val": 50}],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        range_header = request.headers.get("Range")
        start, end = map(int, range_header.split("-"))
        idx = start // page_size
        data = pages[idx] if idx < len(pages) else []
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport, base_url="https://sb.example.com") as client:
        df = await fetch_data_async("trade_logs", page_size=page_size, client=client)

    expected = pd.concat([pd.DataFrame(p) for p in pages], ignore_index=True)
    pd.testing.assert_frame_equal(df, expected)
