"""Backtesting utilities for cointrainer."""

from __future__ import annotations

from .metrics import (
    cagr,
    drawdown_curve,
    hit_rate,
    max_drawdown,
    sharpe,
    sortino,
    summarize,
)
from .optimize import optimize_grid, optimize_optuna
from .run import backtest_csv
from .sim import simulate

__all__ = [
    "backtest_csv",
    "cagr",
    "drawdown_curve",
    "hit_rate",
    "max_drawdown",
    "optimize_grid",
    "optimize_optuna",
    "sharpe",
    "simulate",
    "sortino",
    "summarize",
]
