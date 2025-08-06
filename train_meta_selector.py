from __future__ import annotations

"""Train a meta selector model using LightGBM."""

from typing import Dict, Tuple

import lightgbm as lgb
import pandas as pd


def train_meta_selector(
    X: pd.DataFrame,
    y: pd.Series,
    lgb_params: dict,
    *,
    use_gpu: bool = False,
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """Train a LightGBM model for meta selection.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    lgb_params : dict
        Parameters passed to ``lightgbm.train``.
    use_gpu : bool, optional
        When ``True`` LightGBM runs on GPU by setting ``device='gpu'``.
        Otherwise the model runs on CPU.
    """

    params = dict(lgb_params)
    params["device"] = "gpu" if use_gpu else "cpu"

    dataset = lgb.Dataset(X, label=y)
    booster = lgb.train(params, dataset)

    preds = booster.predict(X)
    if preds.ndim > 1:
        preds = preds.argmax(axis=1)
    metrics = {"accuracy": float((preds == y).mean())}
    return booster, metrics
