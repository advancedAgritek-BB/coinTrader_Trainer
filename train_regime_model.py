from __future__ import annotations

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
