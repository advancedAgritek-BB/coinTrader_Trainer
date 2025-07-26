from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
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

    is_local = os.path.isfile(source_url) or source_url.startswith("file://")

    if is_local:
        if source_url.startswith("file://"):
            source_url = source_url[7:]
        skiprows = 0
        try:
            with open(source_url, "r", encoding="utf-8") as f:
                first_line = f.readline()
            if "cryptodatadownload" in first_line.lower():
                skiprows = 1
        except OSError:
            pass
        df = pd.read_csv(source_url, skiprows=skiprows)
    else:
        response = requests.get(source_url)
        response.raise_for_status()
        if output_file:
            with open(output_file, "wb") as f:
                f.write(response.content)
        first_line = response.text.splitlines()[0].lower()
        skiprows = 1 if "cryptodatadownload" in first_line else 0

        df = pd.read_csv(io.StringIO(response.text), skiprows=skiprows)

    rename_map = {
        "timestamp": "ts",
        "unix": "ts",
        "date": "ts",
        "close": "price",
    }
    rename_map_lower = {k.lower(): v for k, v in rename_map.items()}
    df = df.rename(
        columns={
            col: rename_map_lower[col.lower()]
            for col in df.columns
            if col.lower() in rename_map_lower
        }
    )
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    if "ts" in df.columns and "price" in df.columns:
        df = df.loc[:, ~df.columns.duplicated()]

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
