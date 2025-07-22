"""Evaluation utilities for coinTrader_Trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_signal_pnl(df: pd.DataFrame, preds: np.ndarray) -> float:
    """Simulate PnL based on binary predictions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing either a ``'returns'`` column with asset returns
        or a ``'close'``/``'Close'`` column for computing returns.
    preds : np.ndarray
        Binary predictions of the same length as ``df``. A value of ``1``
        corresponds to a long position, ``0`` to a flat position.

    Returns
    -------
    float
        Annualised Sharpe ratio squared of the resulting strategy.
    """
    if len(df) != len(preds):
        raise ValueError("`preds` length must match `df` length")

    if "returns" in df.columns:
        asset_returns = df["returns"].astype(float)
    elif "close" in df.columns:
        asset_returns = df["close"].astype(float).pct_change().fillna(0.0)
    elif "Close" in df.columns:
        asset_returns = df["Close"].astype(float).pct_change().fillna(0.0)
    else:
        raise ValueError(
            "DataFrame must contain a 'returns', 'close' or 'Close' column"
        )

    preds = np.asarray(preds, dtype=float)
    signals = np.where(preds == 1, 1.0, 0.0)
    strategy_returns = asset_returns * signals

    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std(ddof=0)

    if std_return == 0:
        return 0.0

    sharpe = np.sqrt(365) * (mean_return / std_return)
    return float(sharpe**2)
    """Return annualised Sharpe ratio squared for a simple long-only strategy.

    The strategy assumes unit leverage and goes long when ``preds`` equals
    ``1``. When the prediction is anything else the strategy is flat. The
    input ``df`` should contain a ``Close`` column representing price data from
    which daily returns will be derived.
    """
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must contain a 'Close' column for price data")

    # Calculate daily returns from closing prices
    rets = df['Close'].pct_change().fillna(0.0).to_numpy(float)

    # Determine positions: 1 when long, 0 otherwise
    positions = np.where(preds == 1, 1.0, 0.0)
    positions = positions[: len(rets)]  # align lengths if necessary

    # Strategy PnL assuming +1 leverage
    pnl = rets[: len(positions)] * positions

    # Compute annualised Sharpe ratio
    if pnl.std(ddof=1) == 0:
        return 0.0
    sharpe_daily = pnl.mean() / pnl.std(ddof=1)
    sharpe_annual = sharpe_daily * np.sqrt(365)
    return sharpe_annual**2

