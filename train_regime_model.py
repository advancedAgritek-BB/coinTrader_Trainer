from __future__ import annotations

"""Wrapper around :func:`trainers.regime_lgbm.train_regime_lgbm` with GPU toggle."""

from typing import Dict, Tuple

import pandas as pd
from trainers.regime_lgbm import train_regime_lgbm
from lightgbm import Booster


def train_regime_model(
    X: pd.DataFrame,
    y: pd.Series,
    lgb_params: dict,
    *,
    use_gpu: bool = False,
    **kwargs,
) -> Tuple[Booster, Dict[str, float]]:
    """Train the regime model using LightGBM.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    lgb_params : dict
        Parameters passed to ``train_regime_lgbm``.
    use_gpu : bool, optional
        When ``True`` the ``device`` parameter is forced to ``'gpu'``.
    kwargs : dict
        Additional keyword arguments forwarded to :func:`train_regime_lgbm`.
    """

    params = dict(lgb_params)
    params["device"] = "gpu" if use_gpu else "cpu"
    return train_regime_lgbm(X, y, params, use_gpu=use_gpu, **kwargs)
