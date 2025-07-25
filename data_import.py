from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import Optional

import io
import requests
import pandas as pd
from supabase import create_client

load_dotenv()


def download_historical_data(
    source_url: str,
    symbol: Optional[str] = None,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    *,
    batch_size: int = 1000,  # kept for backwards compatibility
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Download historical trade data from ``source_url``.

    The function fetches ``source_url`` using ``requests`` and parses the
    response as CSV. The optional ``symbol``, ``start_ts`` and ``end_ts``
    parameters are retained for API compatibility but no longer modify the
    request.  Any downloaded data is returned as a ``pandas.DataFrame`` and can
    optionally be saved to ``output_file``.
    """

    response = requests.get(source_url)
    response.raise_for_status()

    if output_file:
        with open(output_file, "wb") as f:
            f.write(response.content)

    df = pd.read_csv(
        io.StringIO(response.text),
        skiprows=1 if "cryptodatadownload" in source_url.lower() else 0,
    )

    return df


def insert_to_supabase(
    df: pd.DataFrame,
    *,
    table: str = "trade_logs",
    batch_size: int = 1000,
) -> None:
    """Insert ``df`` rows into ``table`` using Supabase credentials."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and service key must be set")

    client = create_client(url, key)
    records = df.to_dict(orient="records")
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        client.table(table).insert(chunk).execute()

