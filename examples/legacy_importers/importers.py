"""Utilities for importing and uploading historical trading data."""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from postgrest.exceptions import APIError
from supabase import Client, create_client
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.normalise import normalize_ohlc

load_dotenv()


def download_historical_data(
    source: str,
    *,
    symbol: str | None = None,
    start_ts: str | datetime | None = None,
    end_ts: str | datetime | None = None,
    output_path: str | None = None,
    output_file: str | None = None,
    return_threshold: float = 0.01,
) -> pd.DataFrame:
    """Load historical price data from ``source`` (file path or URL)."""

    is_local = os.path.isfile(source) or source.startswith("file://")
    path = source[7:] if source.startswith("file://") else source

    skiprows = 0
    if is_local:
        try:
            with open(path, encoding="utf-8") as f:
                first_line = f.readline()
            if "cryptodatadownload" in first_line.lower():
                skiprows = 1
        except OSError:
            pass
        df = pd.read_csv(path, skiprows=skiprows)
    else:
        resp = requests.get(source)
        resp.raise_for_status()
        if output_file:
            with open(output_file, "wb") as fh:
                fh.write(resp.content)
        first_line = resp.text.splitlines()[0].lower()
        skiprows = 1 if "cryptodatadownload" in first_line else 0
        df = pd.read_csv(io.StringIO(resp.text), skiprows=skiprows)

    rename_map = {
        "Unix": "unix",
        "unix": "unix",
        "Date": "date",
        "date": "date",
        "Symbol": "symbol",
        "symbol": "symbol",
        "Open": "open",
        "open": "open",
        "High": "high",
        "high": "high",
        "Low": "low",
        "low": "low",
        "Close": "close",
        "close": "close",
        "Volume XRP": "volume_xrp",
        "volume xrp": "volume_xrp",
        "Volume USDT": "volume_usdt",
        "volume usdt": "volume_usdt",
        "tradecount": "tradecount",
        "Trade Count": "tradecount",
    }

    df = df.rename(
        columns={col: rename_map.get(col, rename_map.get(col.lower(), col)) for col in df.columns}
    )

    if df.columns.duplicated().any():
        dupes = list(df.columns[df.columns.duplicated()])
        logging.warning("Dropping duplicate columns: %s", dupes)
        df = df.loc[:, ~df.columns.duplicated()]

    if ("unix" not in df.columns and "date" not in df.columns) or "close" not in df.columns:
        raise ValueError("CSV must contain unix/date timestamp and close columns")

    if "timestamp" not in df.columns:
        if "unix" in df.columns:
            df["timestamp"] = pd.to_datetime(df["unix"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["date"], utc=True)

    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    if start_ts is not None:
        start_ts = pd.to_datetime(start_ts, utc=True)
        df = df[df["timestamp"] >= start_ts]
    if end_ts is not None:
        end_ts = pd.to_datetime(end_ts, utc=True)
        df = df[df["timestamp"] < end_ts]

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    schema_columns = [
        "unix",
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume_xrp",
        "volume_usdt",
        "tradecount",
        "timestamp",
    ]
    df = df[[col for col in schema_columns if col in df.columns]]

    if output_path:
        if output_path.endswith(".parquet"):
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)

    df = normalize_ohlc(df)

    if "target" not in df.columns and "price" in df.columns:
        returns = df["price"].pct_change().shift(-1)
        df["target"] = pd.Series(
            np.where(returns > return_threshold, 1, np.where(returns < -return_threshold, -1, 0)),
            index=df.index,
        ).fillna(0)

    return df


def _get_write_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def _insert_batch(client: Client, table: str, rows: list[dict]) -> None:
    """Insert ``rows`` into ``table`` using ``client`` with retry."""
    client.table(table).insert(rows).execute()


_INSERTED_TABLES: set[tuple[int, str]] = set()


def ensure_table_exists(symbol: str, *, client: Client | None = None) -> str:
    """Create historical prices table for ``symbol`` if needed and return its name."""
    table = f"historical_prices_{symbol.lower()}"
    if client is None:
        client = _get_write_client()
    sql = (
        f"create table if not exists {table} "
        "(like historical_prices including defaults including constraints)"
    )
    try:
        client.rpc("sql", {"query": sql}).execute()
    except APIError as exc:
        if exc.code == "42P01":
            raise RuntimeError("historical_prices table must exist") from exc
        raise
    return table


def insert_to_supabase(
    df: pd.DataFrame,
    url: str | None = None,
    key: str | None = None,
    *,
    table: str | None = None,
    symbol: str | None = None,
    client: Client | None = None,
    batch_size: int = 500,
) -> None:
    """Insert ``df`` rows into Supabase.

    A new Supabase ``Client`` is created from ``url`` and ``key`` if provided.
    Otherwise ``client`` is used or created from environment variables.
    """

    if client is None:
        if url is not None and key is not None:
            client = create_client(url, key)
        elif url is None and key is None:
            client = _get_write_client()
        else:
            raise ValueError("Both url and key must be provided")

    if symbol is not None:
        table_name = f"historical_prices_{symbol.lower()}"
        key_table = (id(client), table_name)
        if key_table not in _INSERTED_TABLES:
            ensure_table_exists(symbol, client=client)
            _INSERTED_TABLES.add(key_table)
        if table is None:
            table = table_name
    elif table is None:
        table = "historical_prices"

    df = df.copy()
    schema_columns = [
        "unix",
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume_xrp",
        "volume_usdt",
        "tradecount",
    ]
    df = df[[col for col in schema_columns if col in df.columns]]

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    records = df.to_dict(orient="records")
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        _insert_batch(client, table, batch)
