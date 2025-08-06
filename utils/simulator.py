"""Simulation utilities for backtesting trading strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.backtest_strategies import backtest


def simulate_trades(market_df: pd.DataFrame, strategies: list) -> pd.DataFrame:
    """Backtest ``strategies`` on ``market_df`` and record trade metrics.

    Parameters
    ----------
    market_df : pd.DataFrame
        Market data used for the backtest.
    strategies : list
        Strategies forwarded to :func:`tools.backtest_strategies.backtest`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing trade results, ``win_rate`` and ``sharpe`` ratio.
    """

    trades_df = backtest(market_df, strategies)
    trades_df["win_rate"] = (trades_df["pnl"] > 0).mean()
    trades_df["sharpe"] = (
        trades_df["pnl"].mean() / trades_df["pnl"].std() * np.sqrt(252)
        if len(trades_df) > 1
        else 0
    )
    trades_df.to_csv("simulated_trades.csv", index=False)
    return trades_df

