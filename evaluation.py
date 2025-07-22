import numpy as np
import pandas as pd


def simulate_signal_pnl(df: pd.DataFrame, preds: np.ndarray) -> float:
    """Return annualised Sharpe ratio squared for a simple long-only strategy.

    The strategy assumes unit leverage and goes long when ``preds`` equals
    ``1``. When the prediction is anything else the strategy is flat. The
    input ``df`` should contain a ``Close`` column representing price data from
    which daily returns will be derived.
    """
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must contain a 'Close' column for price data")

    # Calculate daily returns from closing prices
    rets = df['Close'].pct_change().fillna(0.0).to_numpy(float)

    # Determine positions: 1 when long, 0 otherwise
    positions = np.where(preds == 1, 1.0, 0.0)
    positions = positions[: len(rets)]  # align lengths if necessary

    # Strategy PnL assuming +1 leverage
    pnl = rets[: len(positions)] * positions

    # Compute annualised Sharpe ratio
    if pnl.std(ddof=1) == 0:
        return 0.0
    sharpe_daily = pnl.mean() / pnl.std(ddof=1)
    sharpe_annual = sharpe_daily * np.sqrt(365)
    return sharpe_annual**2

