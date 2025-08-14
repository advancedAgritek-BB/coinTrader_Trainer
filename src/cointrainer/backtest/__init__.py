"""Backtesting helpers for cointrainer."""

from .optimize import optimize_grid, optimize_optuna
from .run import backtest_csv

__all__ = ["optimize_grid", "optimize_optuna", "backtest_csv"]
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
