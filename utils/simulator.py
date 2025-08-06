from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tools.backtest_strategies import backtest


def simulate(df: pd.DataFrame, strategies: Any) -> dict[str, float]:
    """Run a backtest over ``df`` for ``strategies`` and compute metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Market data used for the simulation.
    strategies : Any
        Strategy definitions forwarded to
        :func:`tools.backtest_strategies.backtest`.

    Returns
    -------
    dict[str, float]
        Dictionary containing ``win_rate`` and annualised ``sharpe`` ratio.
    """
    results = backtest(df, strategies)

    if isinstance(results, Mapping):
        pnl = np.asarray(results.get("pnl", []), dtype=float)
    else:
        pnl = np.asarray(results, dtype=float)

    if pnl.size == 0:
        return {"win_rate": 0.0, "sharpe": 0.0}

    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std(ddof=0))
    sharpe = mean_pnl / std_pnl * math.sqrt(252) if std_pnl else 0.0
    win_rate = float((pnl > 0).sum() / pnl.size)

    metrics = {"win_rate": win_rate, "sharpe": sharpe}
    return metrics
