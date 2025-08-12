from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from statsmodels.tsa.stattools import adfuller

try:
    from statsmodels.tsa.stattools import hurst as _hurst
except ImportError:  # pragma: no cover - statsmodels may lack hurst
    _hurst = None

def _fallback_hurst(series: np.ndarray) -> float:
    """Estimate Hurst exponent using a simple rescaled range approach."""
    lags = np.arange(2, min(100, len(series) // 2))
    if len(lags) == 0:
        return 0.5
    tau = [np.sqrt(np.std(series[lag:] - series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def train_mean_reversion(csv_path: str) -> dict[str, float]:
    """Train a mean reversion model based on an Ornstein-Uhlenbeck process.

    Parameters
    ----------
    csv_path: str
        Path to a CSV file containing at least a ``close`` column.

    Returns
    -------
    Dict[str, float]
        Dictionary with fitted parameters and diagnostic statistics.

    Raises
    ------
    ValueError
        If the ADF test fails to reject the unit root hypothesis.
    """
    data = pd.read_csv(csv_path)
    prices = data["close"].astype(float).values

    def residuals(params: np.ndarray) -> np.ndarray:
        theta, mu = params
        dx = np.diff(prices)
        model_dx = theta * (mu - prices[:-1])
        return dx - model_dx

    initial = np.array([0.1, prices.mean()])
    result = least_squares(residuals, initial)
    theta, mu = result.x
    sigma = np.std(result.fun)

    adf_stat, adf_pvalue, *_ = adfuller(prices)
    if _hurst is not None:
        hurst_exp = _hurst(prices)
    else:  # fallback if statsmodels lacks hurst
        hurst_exp = _fallback_hurst(prices)

    if adf_pvalue >= 0.05:
        raise ValueError(f"Series is not mean-reverting (ADF p-value={adf_pvalue:.4f})")

    params = np.array([theta, mu, sigma])
    out_path = Path(__file__).with_name("ou_params.npy")
    np.save(out_path, params)

    return {
        "theta": float(theta),
        "mu": float(mu),
        "sigma": float(sigma),
        "adf_pvalue": float(adf_pvalue),
        "hurst": float(hurst_exp),
    }
