from __future__ import annotations

"""Fetch OHLC data from a CCXT exchange and append it to Supabase."""

import argparse
import os
import time
from typing import Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
DEFAULT_TABLE = os.environ.get("CCXT_TABLE", "ohlc_data")


def get_exchange(name: str) -> ccxt.Exchange:
    """Instantiate a CCXT exchange by name."""
    if not hasattr(ccxt, name):
        raise ValueError(f"Exchange '{name}' not found in ccxt")
    return getattr(ccxt, name)()


def get_markets(exchange: ccxt.Exchange) -> list[str]:
    """Return list of available market symbols for ``exchange``."""
    markets = exchange.load_markets()
    return list(markets.keys())


def get_last_ts(client: Client, symbol: str, table: str) -> Optional[int]:
    """Get the Unix timestamp of the latest entry for ``symbol`` in ``table``."""
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


def fetch_ccxt_ohlc(exchange: ccxt.Exchange, symbol: str, timeframe: str = "1m") -> pd.DataFrame:
    """Fetch recent OHLCV data for ``symbol`` from ``exchange``."""
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=720)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["exchange"] = exchange.id
    df["price"] = df["close"]
    numeric_cols = ["open", "high", "low", "close", "price", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    return df


def insert_to_supabase(client: Client, df: pd.DataFrame, *, table: str, batch_size: int = 1000) -> None:
    """Insert DataFrame rows to Supabase in batches."""
    records = df.to_dict(orient="records")
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        client.table(table).insert(chunk).execute()


def append_ccxt_data_all(
    exchange_name: str = "binance",
    timeframe: str = "1m",
    delay_sec: float = 1.0,
    *,
    table: str = DEFAULT_TABLE,
) -> None:
    """Fetch and append OHLC data for all markets on an exchange."""
    exchange = get_exchange(exchange_name)
    markets = get_markets(exchange)
    ex_id = getattr(exchange, "id", exchange_name)
    for symbol in markets:
        last_ts = get_last_ts(client, symbol, table)
        df = fetch_ccxt_ohlc(exchange, symbol, timeframe)
        if last_ts is not None:
            last_dt = pd.to_datetime(last_ts, unit="s", utc=True)
            df = df[df["ts"] > last_dt]
        if not df.empty:
            insert_to_supabase(client, df, table=table)
            print(f"Appended {len(df)} rows for {ex_id}:{symbol}")
        time.sleep(delay_sec)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLC data from a CCXT exchange")
    parser.add_argument(
        "--exchange",
        default=os.environ.get("CCXT_EXCHANGE", "binance"),
        help="Exchange id, e.g. binance",
    )
    parser.add_argument(
        "--table",
        default=os.environ.get("CCXT_TABLE", "ohlc_data"),
        help="Supabase table for inserted rows",
    )
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--delay-sec", type=float, default=1.0)

    args = parser.parse_args(argv)
    append_ccxt_data_all(
        exchange_name=args.exchange,
        timeframe=args.timeframe,
        delay_sec=args.delay_sec,
        table=args.table,
    )


if __name__ == "__main__":
    main()
