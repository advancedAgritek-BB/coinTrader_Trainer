import numpy as np
import pandas as pd


def train(csv_path: str, alpha: float = 1.0) -> dict:
    """Compute upper-confidence-bound means from PnL logs.

    Parameters
    ----------
    csv_path : str
        Path to a ``pnl.csv`` file with columns ``regime``, ``strategy`` and ``pnl``.
    alpha : float, optional
        Exploration parameter scaling the standard deviation term.

    Returns
    -------
    dict
        Mapping of ``(regime, strategy)`` to UCB-adjusted mean PnL.
    """
    df = pd.read_csv(csv_path)
    grouped = df.groupby(["regime", "strategy"])["pnl"]
    means = grouped.mean()
    stds = grouped.std().fillna(0.0)
    counts = grouped.count()
    ucb = means + alpha * stds / np.sqrt(counts)
    ucb_dict = ucb.to_dict()
    np.save("bandit_means.npy", ucb_dict)
    return ucb_dict
