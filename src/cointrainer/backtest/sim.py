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
import numpy as np
import pandas as pd
from typing import Literal, Callable, Dict
from .metrics import summarize

def simulate(
    prices: pd.Series,
    position: pd.Series | np.ndarray,
    fee_bps: float = 2.0,          # per side, in basis points
    slip_bps: float = 0.0,         # slippage per trade, bps
    periods_per_year: float = 525600.0,  # 1-min bars
) -> Dict:
    """
    Vectorized PnL simulation:
      - position in [-1..1], same length as prices
      - fees charged when position changes (turnover)
      - returns are close-to-close
    """
    position = pd.Series(position, index=prices.index).astype(float)
    ret = prices.pct_change().fillna(0.0)
    pos_prev = position.shift(1).fillna(0.0)
    turnover = (position - pos_prev).abs()
    fee = (fee_bps / 1e4) * turnover
    slip = (slip_bps / 1e4) * turnover
    gross = position * ret
    net = gross - fee - slip
    equity = (1.0 + net).cumprod()
    trades_idx = turnover[turnover > 0.0].index
    stats = summarize(equity, net, trades_idx, periods_per_year)
    return {"equity": equity, "net": net, "turnover": turnover, "stats": stats}
