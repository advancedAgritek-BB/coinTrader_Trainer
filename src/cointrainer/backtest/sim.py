from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

__all__ = ["simulate"]

def simulate(
    prices: pd.Series,
    positions: np.ndarray,
    *,
    fee_bps: float = 2.0,
    slip_bps: float = 0.0,
) -> Dict[str, Any]:
    """Simulate equity curve from price series and positions.

    Parameters
    ----------
    prices : pandas.Series
        Price series aligned to ``positions``.
    positions : np.ndarray
        Array of position values (can be fractional).
    fee_bps : float
        Transaction fee in basis points applied on position changes.
    slip_bps : float
        Slippage in basis points applied on position changes.
    """

    prices = prices.astype(float)
    ret = prices.pct_change().fillna(0.0).to_numpy()
    pos = np.asarray(positions, dtype=float)
    if len(ret) != len(pos):
        raise ValueError("positions length must match prices")

    pnl = ret * pos
    cost = (fee_bps + slip_bps) / 10000.0
    pos_diff = np.diff(np.concatenate([[0.0], pos]))
    pnl -= cost * np.abs(pos_diff)
    equity = pd.Series((1.0 + pnl).cumprod(), index=prices.index)

    mean = pnl.mean()
    std = pnl.std(ddof=0)
    sharpe = float(np.sqrt(365) * mean / std) if std > 0 else 0.0
    peaks = equity.cummax()
    drawdown = (equity / peaks) - 1.0
    max_dd = float(drawdown.min())
    stats = {
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
    return {"equity": equity, "stats": stats}
