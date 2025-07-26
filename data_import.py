from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from supabase import Client, create_client
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


def download_historical_data(
    source_url: str,
    symbol: Optional[str] = None,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    *,
    output_file: Optional[str] = None,
    return_threshold: float = 0.01,
) -> pd.DataFrame:
    """Download historical trade data from ``source_url``.

    ``symbol``, ``start_ts`` and ``end_ts`` are retained for API compatibility
    but do not affect the download.  Columns are normalised according to the
    rename map and timestamps are converted to ISO format in a ``ts`` column.
    If ``target`` is missing and a ``price`` column is present, it will be
    generated using ``return_threshold`` to classify price changes.
    """

    is_local = os.path.isfile(source_url) or source_url.startswith("file://")
    path = source_url[7:] if source_url.startswith("file://") else source_url

    lower = path.lower()
    skiprows = 1 if ("binance" in lower or "cryptodatadownload" in lower) else 0

    if is_local:
        df = pd.read_csv(path, skiprows=skiprows)
    else:
        resp = requests.get(source_url)
        resp.raise_for_status()
        if output_file:
            with open(output_file, "wb") as fh:
                fh.write(resp.content)
        first_line = resp.text.splitlines()[0].lower()
        skiprows = 1 if "cryptodatadownload" in first_line else 0
        df = pd.read_csv(io.StringIO(resp.text), skiprows=skiprows)

    rename_map = {
        "timestamp": "ts",
        "unix": "ts",
        "date": "ts",
        "time": "ts",
        "close": "price",
        "price": "price",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "volume usdt": "volume",
    }
    rename_map_lower = {k.lower(): v for k, v in rename_map.items()}
    df = df.rename(
        columns={
            col: rename_map_lower[col.lower()]
            for col in df.columns
            if col.lower() in rename_map_lower
        }
    )

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]

    if start_ts is not None and "ts" in df.columns:
        start = pd.to_datetime(start_ts, utc=True)
        df = df[df["ts"] >= start]
    if end_ts is not None and "ts" in df.columns:
        end = pd.to_datetime(end_ts, utc=True)
        df = df[df["ts"] < end]

    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)

    if "target" not in df.columns and "price" in df.columns:
        returns = df["price"].pct_change().shift(-1)
        df["target"] = pd.Series(
            np.where(
                returns > return_threshold,
                1,
                np.where(returns < -return_threshold, -1, 0),
            ),
            index=df.index,
        ).fillna(0)

    return df


def _get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and service key must be set")
    return create_client(url, key)


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def _insert_batch(client: Client, table: str, rows: list[dict]) -> None:
    client.table(table).insert(rows).execute()


def insert_to_supabase(
    df: pd.DataFrame,
    *,
    table: str = "ohlc_data",
    batch_size: int = 1000,
) -> None:
    """Insert ``df`` rows into ``table`` using Supabase credentials."""
    client = _get_client()

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    records = df.to_dict(orient="records")
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        _insert_batch(client, table, chunk)
