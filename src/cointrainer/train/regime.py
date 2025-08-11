from __future__ import annotations

"""Train a LightGBM-based regime classifier and persist the artifact."""

from pathlib import Path
from typing import Any

import argparse
import pickle
from datetime import datetime

import lightgbm as lgb
import pandas as pd

from cointrainer.registry import save_model


def _load_data(data: pd.DataFrame | str) -> pd.DataFrame:
    """Return ``data`` as a DataFrame.

    Parameters
    ----------
    data:
        Either a DataFrame or path to a CSV file.
    """

    if isinstance(data, pd.DataFrame):
        return data
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split ``df`` into features ``X`` and target ``y``.

    The helper selects numeric columns and separates the column named
    ``high_vol`` or ``target`` if present. Otherwise the last numeric column is
    treated as the target.
    """

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
    registry: Any | None = None,
    symbol: str = "BTCUSDT",
    horizon: str = "15m",
    thresholds: dict | None = None,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier for market regimes and save it."""

    df = _load_data(data)
    X, y = _prepare_xy(df)

    params: dict[str, Any] = {"objective": "binary"}
    if use_gpu:
        params["device"] = "gpu"

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    key = f"models/regime/{symbol}/{ts}_regime_lgbm.pkl"
    blob = pickle.dumps(model)
    metadata = {
        "feature_list": list(X.columns),
        "label_order": [-1, 0, 1],
        "horizon": horizon,
    }
    if thresholds:
        metadata["thresholds"] = thresholds
    save_model(key, blob, metadata)
    return model


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for training the regime model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Path to CSV containing training data")
    parser.add_argument(
        "--model-name", default="regime_model", help="Name used when uploading"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args(argv)

    train_regime_model(args.data, use_gpu=not args.cpu, model_name=args.model_name)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["train_regime_model", "main"]

