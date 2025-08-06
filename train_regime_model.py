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
"""Train a high-volatility regime classifier using LightGBM."""

from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from registry import ModelRegistry


def _load_data(data: pd.DataFrame | str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    numeric = df.select_dtypes("number").copy()
    if "high_vol" in numeric.columns:
        y = numeric.pop("high_vol")
    elif "target" in numeric.columns:
        y = numeric.pop("target")
    else:
        y = numeric.iloc[:, -1]
        numeric = numeric.iloc[:, :-1]
    return numeric, y


def train_regime_model(
    data: pd.DataFrame | str,
    *,
    use_gpu: bool = True,
    model_name: str = "regime_model",
    registry: ModelRegistry | None = None,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier for identifying high-volatility regimes."""
    df = _load_data(data)
    X, y = _prepare_xy(df)

    params: dict[str, Any] = {"objective": "binary"}
    if use_gpu:
        params["device"] = "gpu"

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    if registry is None:
        registry = ModelRegistry()
    registry.upload(model, model_name)
    return model


__all__ = ["train_regime_model"]
