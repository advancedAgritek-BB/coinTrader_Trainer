from __future__ import annotations
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
