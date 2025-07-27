"""
Helpers for fetching OHLC data from Kraken and appending it to Supabase.

How to Use:
Run `python kraken_fetch.py` manually or via cron, e.g., every 6 hours:
0 */6 * * * python /path/to/kraken_fetch.py
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import Client, create_client
import argparse

load_dotenv()

logger = logging.getLogger(__name__)

# Supabase client (using service key for writes)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Destination table for new rows (override with KRAKEN_TABLE or --table)
DEFAULT_TABLE = os.environ.get("KRAKEN_TABLE", "trade_logs")

def get_tradable_pairs() -> list[str]:
    """Fetch list of all tradable asset pairs from Kraken."""
    resp = requests.get("https://api.kraken.com/0/public/AssetPairs")
    resp.raise_for_status()
    data = resp.json()
    if "error" in data and data["error"]:
        raise ValueError(data["error"])
    return list(data["result"].keys())  # e.g., ['XBTUSD', 'ETHUSD']

def get_last_ts(client: Client, symbol: str, table: str) -> Optional[int]:
    """Get the Unix timestamp of the latest entry for a symbol (or None if empty)."""
    resp = (
        client.table(table)
        .select("ts")
        .eq("symbol", symbol)
        .order("ts", desc=True)
        .limit(1)
        .execute()
    )
    if resp.data:
        return int(pd.to_datetime(resp.data[0]["ts"]).timestamp())
    return None

def fetch_kraken_ohlc(pair: str, interval: int = 1) -> pd.DataFrame:
    """Fetch recent OHLC for a pair (last ~720 candles)."""
    params = {"pair": pair, "interval": interval}
    resp = requests.get("https://api.kraken.com/0/public/OHLC", params=params)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data and data["error"]:
        raise ValueError(data["error"])
    ohlc_data = data["result"][pair]
    df = pd.DataFrame(
        ohlc_data,
        columns=["ts", "open", "high", "low", "close", "vwap", "volume", "trades"],
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    if not df["ts"].diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        logger.warning("Gaps detected in OHLC")
    df["symbol"] = pair
    df["price"] = df["close"]  # Alias for code compatibility
    # Convert types
    numeric_cols = ["open", "high", "low", "close", "price", "vwap", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    df["trades"] = df["trades"].astype(int)
    return df

def insert_to_supabase(
    client: Client,
    df: pd.DataFrame,
    *,
    table: str,
    batch_size: int = 1000,
    conflict_cols: tuple[str, ...] = ("ts", "symbol"),
) -> None:
    """Insert or update DataFrame rows to Supabase (batches for efficiency)."""
    records = df.to_dict(orient="records")
    on_conflict = ",".join(conflict_cols)
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        client.table(table).upsert(chunk, on_conflict=on_conflict).execute()  # UNIQUE constraint prevents duplicates

def append_kraken_data(
    interval: int = 1, delay_sec: float = 1.0, *, table: str = DEFAULT_TABLE
) -> None:
    """Fetch and append recent OHLC for all pairs (filter to new data only)."""
    pairs = get_tradable_pairs()
    for pair in pairs:
        last_ts = get_last_ts(client, pair, table)
        df = fetch_kraken_ohlc(pair, interval)
        if last_ts is not None:
            last_dt = pd.to_datetime(last_ts, unit="s", utc=True)
            df = df[df["ts"] > last_dt]
        if not df.empty:
            insert_to_supabase(client, df, table=table)
            print(f"Appended {len(df)} rows for {pair}")
        time.sleep(delay_sec)

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLC data from Kraken")
    parser.add_argument(
        "--table",
        default=os.environ.get("KRAKEN_TABLE", "trade_logs"),
        help="Supabase table for inserted rows",
    )
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--delay-sec", type=float, default=1.0)

    args = parser.parse_args(argv)
    append_kraken_data(interval=args.interval, delay_sec=args.delay_sec, table=args.table)


if __name__ == "__main__":
    main()
