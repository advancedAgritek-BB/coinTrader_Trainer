"""Backtesting utilities for simple strategy evaluation.

The heavy dependencies required for backtesting are imported lazily to
keep module import cheap during test runs.
"""

from __future__ import annotations


def backtest(*args, **kwargs):  # pragma: no cover - thin wrapper
    from .backtest import backtest as _backtest

    return _backtest(*args, **kwargs)


__all__ = ["backtest"]
