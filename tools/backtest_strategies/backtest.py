"""Simple backtesting utilities for multiple strategies.

This module provides :func:`backtest` which simulates trading for a list of
strategies.  Strategies are identified by name and mapped to functions from the
``crypto_bot`` package.  For each strategy the function loops through the
provided DataFrame calling the appropriate prediction function to generate
signals.  Trades are executed immediately at the ``close`` price of each row
including a fixed 0.1% fee on entry and exit.  Closed trades are saved to
``simulated_trades.csv``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

# Import strategy modules.  These modules are expected to exist within the
# ``crypto_bot`` package.  Each module exposes a callable used below.
from crypto_bot.lstm import lstm_bot
from crypto_bot.mean_rev import mean_bot
from crypto_bot.ml.lgbm import ml_signal_model

FEE_RATE = 0.001  # 0.1%


@dataclass
class TradeRecord:
    """Container for a single completed trade."""

    strategy: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float


def _get_strategy_function(name: str) -> Callable[[pd.Series], float]:
    """Map a strategy name to its prediction function.

    Parameters
    ----------
    name:
        Short name of the strategy. Supported values are ``"lstm"``, ``"mean"``
        and ``"ml"``.
    """

    mapping: dict[str, Callable[[pd.Series], float]] = {
        "lstm": lstm_bot.predict,
        "mean": mean_bot.score,
        "ml": ml_signal_model.get_signal,
    }
    if name not in mapping:
        raise ValueError(f"Unknown strategy: {name}")
    return mapping[name]


def backtest(df: pd.DataFrame, strategies_list: list[str]) -> pd.DataFrame:
    """Run a very small backtest for the given strategies.

    The DataFrame ``df`` must contain a ``'close'`` column and be indexed by
    timestamps.  For each strategy in ``strategies_list`` the corresponding
    prediction function is called for every row to generate trading signals.
    Signals greater than zero open/maintain a long position, signals below zero
    open/maintain a short position and a zero signal closes any open position.

    A fixed fee of 0.1% is applied to both entry and exit prices.  Completed
    trades are written to ``simulated_trades.csv`` and returned as a DataFrame.
    """

    trades: list[TradeRecord] = []

    for strat in strategies_list:
        func = _get_strategy_function(strat)

        position = 0  # 1 for long, -1 for short, 0 flat
        entry_price = 0.0
        entry_time: pd.Timestamp | None = None

        for ts, row in df.iterrows():
            signal = func(row)
            price = float(row["close"])

            if position == 0:
                if signal > 0:
                    entry_price = price * (1 + FEE_RATE)
                    entry_time = ts
                    position = 1
                elif signal < 0:
                    entry_price = price * (1 - FEE_RATE)
                    entry_time = ts
                    position = -1
                continue

            if position == 1 and signal <= 0:
                exit_price = price * (1 - FEE_RATE)
                pnl = exit_price - entry_price
                trades.append(
                    TradeRecord(
                        strategy=strat,
                        side="long",
                        entry_time=entry_time,
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                    )
                )
                position = 0
                if signal < 0:
                    entry_price = price * (1 - FEE_RATE)
                    entry_time = ts
                    position = -1
            elif position == -1 and signal >= 0:
                exit_price = price * (1 + FEE_RATE)
                pnl = entry_price - exit_price
                trades.append(
                    TradeRecord(
                        strategy=strat,
                        side="short",
                        entry_time=entry_time,
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                    )
                )
                position = 0
                if signal > 0:
                    entry_price = price * (1 + FEE_RATE)
                    entry_time = ts
                    position = 1

        # Close any remaining position at the final price
        if position != 0 and entry_time is not None:
            price = float(df["close"].iloc[-1])
            ts = df.index[-1]
            if position == 1:
                exit_price = price * (1 - FEE_RATE)
                pnl = exit_price - entry_price
                trades.append(
                    TradeRecord(
                        strategy=strat,
                        side="long",
                        entry_time=entry_time,
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                    )
                )
            else:
                exit_price = price * (1 + FEE_RATE)
                pnl = entry_price - exit_price
                trades.append(
                    TradeRecord(
                        strategy=strat,
                        side="short",
                        entry_time=entry_time,
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                    )
                )

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    trades_df.to_csv("simulated_trades.csv", index=False)
    return trades_df


__all__ = ["backtest"]
