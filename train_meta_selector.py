from __future__ import annotations

"""Train a meta selector model using LightGBM."""

from typing import Dict, Tuple
"""Train a LightGBM meta-selector on simulator output and upload to Supabase."""

from pathlib import Path
from typing import Any

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
from registry import ModelRegistry


def _load_data(data: pd.DataFrame | str) -> pd.DataFrame:
    """Return ``data`` as a DataFrame.

    Parameters
    ----------
    data:
        Either a DataFrame with simulator output or a path to a CSV file.
    """
    if isinstance(data, pd.DataFrame):
        return data
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build features from simulator output.

    This helper selects numeric columns and separates the target column named
    ``target`` (if present) or the last numeric column otherwise.
    """
    numeric = df.select_dtypes("number").copy()
    if numeric.empty:
        raise ValueError("No numeric columns found for feature generation")

    if "target" in numeric.columns:
        y = numeric.pop("target")
    else:
        y = numeric.iloc[:, -1]
        numeric = numeric.iloc[:, :-1]
    return numeric, y


def train_meta_selector(
    data: pd.DataFrame | str,
    *,
    use_gpu: bool = False,
    model_name: str = "meta_selector",
    registry: ModelRegistry | None = None,
) -> lgb.LGBMClassifier:
    """Train a LightGBM meta-selector and upload the model to Supabase.

    Parameters
    ----------
    data:
        DataFrame or path to CSV containing simulator output. The dataset must
        contain a numeric ``target`` column or the last numeric column will be
        treated as the target.
    use_gpu:
        If ``True`` enable LightGBM GPU acceleration. Defaults to ``False``.
    model_name:
        Name used when storing the model in Supabase. Defaults to
        ``"meta_selector"``.
    registry:
        Optional :class:`ModelRegistry` instance. If not provided a new
        instance is created using environment configuration.
    """
    df = _load_data(data)
    X, y = _build_features(df)

    params: dict[str, Any] = {"objective": "binary"}
    if use_gpu:
        params["device"] = "gpu"

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    if registry is None:
        registry = ModelRegistry()
    registry.upload(model, model_name)
    return model


__all__ = ["train_meta_selector"]
