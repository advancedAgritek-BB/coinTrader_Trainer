"""Evaluation utilities for coinTrader_Trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_signal_pnl(
    df: pd.DataFrame,
    preds: np.ndarray,
    costs: float = 0.002,
    slippage: float = 0.005,
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
        Transaction cost applied when positions change. Defaults to ``0.002``.
    slippage : float, optional
        Price slippage incurred when positions change. The deduction is
        proportional to the absolute change in signal. Defaults to ``0.005``.

    Returns
    -------
    dict
        Dictionary with strategy performance metrics:

        ``sharpe_squared``
            Square of the Sharpe ratio.
        ``sharpe``
            Annualised Sharpe ratio of the strategy.
        ``sortino``
            Annualised Sortino ratio of the strategy.
        ``max_drawdown``
            Maximum drawdown over the evaluated period.
        ``win_rate``
            Fraction of profitable trades.
        ``calmar_ratio``
            Annualised return divided by ``max_drawdown``.
        ``profit_factor``
            Ratio of gross profits to gross losses.
        Dictionary containing metrics for the simulated strategy including
        ``'sharpe_squared'``, ``'sharpe'``, ``'sortino'``, ``'max_drawdown'``,
        ``'win_rate'``, ``'calmar_ratio'`` and ``'profit_factor'``.
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
    strategy_returns -= slippage * np.abs(signal_diff)
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

    cum_returns = (1.0 + strategy_returns).cumprod()
    peaks = cum_returns.cummax()
    drawdown = cum_returns / peaks - 1.0
    max_drawdown = drawdown.min()

    trade_returns = []
    current_signal = 0.0
    current_ret = 0.0
    for ret, sig in zip(strategy_returns, signals):
        if sig != current_signal:
            if current_signal != 0.0:
                trade_returns.append(current_ret)
                current_ret = 0.0
            current_signal = sig
        if sig != 0.0:
            current_ret += ret
    if current_signal != 0.0:
        trade_returns.append(current_ret)

    wins = [tr for tr in trade_returns if tr > 0]
    losses = [tr for tr in trade_returns if tr < 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    annual_return = strategy_returns.mean() * 365
    calmar_ratio = (
        float(annual_return) / abs(max_drawdown) if max_drawdown != 0 else 0.0
    )

    gains = strategy_returns[strategy_returns > 0].sum()
    losses = -strategy_returns[strategy_returns < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else 0.0

    metrics = {
        "sharpe_squared": float(sharpe**2),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "calmar_ratio": float(calmar_ratio),
        "profit_factor": float(profit_factor),
    }

    return metrics


def run_backtest(
    df: pd.DataFrame,
    preds: np.ndarray,
    *,
    cash: float = 10000.0,
    commission: float = 0.001,
) -> float:
    """Execute a simple Backtrader backtest using prediction signals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns consumed by ``backtrader``.
    preds : np.ndarray
        Array of trading signals with 1 for long, -1 for short and 0 for flat.
    cash : float, optional
        Starting portfolio value. Defaults to ``10000.0``.
    commission : float, optional
        Commission rate per trade. Defaults to ``0.001``.

    Returns
    -------
    float
        Final broker value after running the backtest.
    """

    import backtrader as bt

    class _SignalStrategy(bt.Strategy):
        params = dict(signals=None)

        def __init__(self):
            self._idx = 0

        def next(self):
            if self._idx >= len(self.p.signals):
                return
            sig = self.p.signals[self._idx]
            self._idx += 1

            pos = self.position.size
            if sig == 1 and pos <= 0:
                if pos < 0:
                    self.close()
                self.buy()
            elif sig == -1 and pos >= 0:
                if pos > 0:
                    self.close()
                self.sell()
            elif sig == 0 and pos != 0:
                self.close()

    data = bt.feeds.PandasData(dataname=df)
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(_SignalStrategy, signals=list(preds))
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.run()
    return float(cerebro.broker.getvalue())


from backtest import run_backtest as bt_run


def full_strategy_eval(
    df: pd.DataFrame,
    preds: np.ndarray,
    **bt_kwargs,
) -> dict:
    """Evaluate strategy PnL and final portfolio value via backtesting.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data indexed by datetime.
    preds : np.ndarray
        Array of trade signals aligned to ``df``.
    **bt_kwargs : dict
        Additional keyword arguments forwarded to :func:`backtest.run_backtest`.

    Returns
    -------
    dict
        ``simulate_signal_pnl`` metrics with ``final_portfolio_value`` added.
    """

    slippage = bt_kwargs.setdefault("slippage", 0.005)
    costs = bt_kwargs.setdefault("costs", 0.002)

    pnl_metrics = simulate_signal_pnl(df, preds, costs=costs, slippage=slippage)
    calmar_ratio = float(pnl_metrics.get("calmar_ratio", 0.0))

    stats = bt_run(df, preds, **bt_kwargs)
    if isinstance(stats, dict):
        final_val = float(stats.get("final_value", 0.0))
    else:
        final_val = float(stats)

    metrics = dict(pnl_metrics)
    metrics["calmar_ratio"] = calmar_ratio
    metrics["final_portfolio_value"] = final_val
    return metrics
