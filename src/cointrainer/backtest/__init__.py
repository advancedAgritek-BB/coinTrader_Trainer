"""Backtesting utilities for cointrainer."""

from .sim import simulate
from .metrics import (
    drawdown_curve,
    max_drawdown,
    sharpe,
    sortino,
    cagr,
    hit_rate,
    summarize,
)

__all__ = [
    "simulate",
    "drawdown_curve",
    "max_drawdown",
    "sharpe",
    "sortino",
    "cagr",
    "hit_rate",
    "summarize",
]
