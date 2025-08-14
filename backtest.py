"""Backtesting utilities for coinTrader_Trainer.

This module provides a thin wrapper around Backtrader to quickly
simulate trading strategies from Pandas DataFrames.  The main entry
point is :func:`run_backtest` which executes a simple strategy based on
precomputed trade signals.

Example
-------
>>> df = pd.DataFrame({"close": [1, 2, 3]})
>>> signals = [1, 0, -1]
>>> stats = run_backtest(df, signals)
>>> isinstance(stats["final_value"], (int, float))
True
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import backtrader as bt
import pandas as pd


class CryptoStrategy(bt.Strategy):
    """Execute trades according to a list/array of signals.

    Parameters
    ----------
    signals : Iterable[int | float]
        Sequence of trading signals aligned with the data feed. ``1``
        represents a long position, ``-1`` a short position and ``0``
        means flat. This parameter is required and must not be
        ``None``.
    """

    params: ClassVar[dict[str, Any]] = {"signals": None}

    def __init__(self) -> None:
        if self.params.signals is None:
            msg = "signals parameter is required"
            raise ValueError(msg)
        self.signals = list(self.params.signals)
        self._idx = 0

    def next(self) -> None:
        if self._idx >= len(self.signals):
            return

        signal = self.signals[self._idx]
        pos = self.position.size

        if signal > 0:  # long
            if pos <= 0:
                if pos < 0:
                    self.close()
                self.buy(size=1)
        elif signal < 0:  # short
            if pos >= 0:
                if pos > 0:
                    self.close()
                self.sell(size=1)
        else:  # flat
            if pos != 0:
                self.close()

        self._idx += 1


def run_backtest(
    df: pd.DataFrame,
    signals: Iterable[int | float],
    slippage: float = 0.005,
    costs: float = 0.002,
) -> dict[str, Any]:
    """Run a simple Backtrader backtest over ``df`` using ``signals``.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLCV data indexed by datetime or containing a ``'close'`` column.
    signals : Iterable[int | float]
        Sequence of trade signals aligned to ``df``. ``1`` represents a
        long position, ``-1`` a short position and ``0`` flat. This
        parameter is required and must not be ``None``.
    slippage : float, optional
        Percentage slippage applied to trades. Defaults to ``0.005``.
    costs : float, optional
        Commission percentage for trades. Defaults to ``0.002``.

    Returns
    -------
    dict
        Dictionary including at least ``'start_value'`` and
        ``'final_value'`` of the portfolio.
    """

    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(CryptoStrategy, signals=list(signals))
    cerebro.broker.setcommission(commission=costs)
    cerebro.broker.set_slippage_perc(slippage)

    start_value = cerebro.broker.getvalue()
    cerebro.run()
    final_value = cerebro.broker.getvalue()

    return {"start_value": start_value, "final_value": final_value}


