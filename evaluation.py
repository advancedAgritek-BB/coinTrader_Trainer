"""Evaluation utilities for coinTrader_Trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_signal_pnl(
    df: pd.DataFrame, preds: np.ndarray, costs: float = 0.001
) -> dict:
    """Simulate PnL for long/short/flat predictions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing either a ``'returns'`` column with asset returns
        or a ``'close'``/``'Close'`` column for computing returns.
    preds : np.ndarray
        Array of predictions of the same length as ``df``. ``1`` denotes a
        long position, ``-1`` a short position and ``0`` a flat position.
    costs : float, optional
        Transaction cost applied when positions change. Defaults to ``0.001``.

    Returns
    -------
    dict
        Dictionary containing ``'sharpe_squared'``, ``'sharpe'`` and ``
        'sortino'`` ratios for the strategy.
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
    signals = np.where(preds == 1, 1.0, np.where(preds == -1, -1.0, 0.0))
    strategy_returns = asset_returns * signals

    signal_diff = np.diff(signals, prepend=0)
    strategy_returns -= costs * np.abs(signal_diff)

    strategy_std = strategy_returns.std(ddof=0)
    if strategy_std == 0 or np.isnan(strategy_std):
        sharpe = 0.0
    else:
        sharpe = np.sqrt(365) * (strategy_returns.mean() / strategy_std)

    downside_std = strategy_returns[strategy_returns < 0].std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        sortino = 0.0
    else:
        sortino = np.sqrt(365) * (strategy_returns.mean() / downside_std)

    return {
        "sharpe_squared": float(sharpe**2),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
    }

