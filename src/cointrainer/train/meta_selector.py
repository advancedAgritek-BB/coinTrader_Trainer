from __future__ import annotations

"""Train the LightGBM-based meta selector and store it in Supabase."""

import argparse
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from cointrainer.registry import ModelRegistry


def _load_data(data: pd.DataFrame | str) -> pd.DataFrame:
    """Return ``data`` as a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    data:
        DataFrame with simulator output or path to a CSV file.
    """
    if isinstance(data, pd.DataFrame):
        return data
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate numeric features and target column from ``df``."""
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
    """Train a LightGBM meta-selector and upload the model to Supabase."""
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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="CSV file with simulator output")
    parser.add_argument("--model-name", default="meta_selector", help="Name for the stored model")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU training")
    args = parser.parse_args()
    train_meta_selector(args.data, use_gpu=args.gpu, model_name=args.model_name)


__all__ = ["main", "train_meta_selector"]


if __name__ == "__main__":  # pragma: no cover - script mode
    main()
