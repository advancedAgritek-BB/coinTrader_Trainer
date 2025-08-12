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
from cointrainer.features.build import build_features, make_labels


def _load_data(data: pd.DataFrame | str) -> pd.DataFrame:
    """Return ``data`` as a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def train_regime_model(
    data: pd.DataFrame | str,
    *,
    use: dict | None = None,
    params: dict | None = None,
    horizon: str = "1",
    thresholds: dict | None = None,
    use_gpu: bool = True,
    model_name: str = "regime_model",
    registry: Any | None = None,
    symbol: str = "BTCUSDT",
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier for market regimes and save it."""

    df = _load_data(data)
    X, feat_meta = build_features(df, use=use or {}, params=params or {})
    y, label_meta = make_labels(df, horizon=horizon, thresholds=thresholds or {})
    X, y = X.align(y, join="inner")

    lgb_params: dict[str, Any] = {"objective": "binary"}
    if use_gpu:
        lgb_params["device"] = "gpu"

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X, y)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    key = f"models/regime/{symbol}/{ts}_regime_lgbm.pkl"
    blob = pickle.dumps(model)
    metadata = {**feat_meta, **label_meta}
    save_model(key, blob, metadata)
    return model


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for training the regime model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Path to CSV containing training data")
    parser.add_argument("--model-name", default="regime_model", help="Name used when uploading")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args(argv)

    train_regime_model(args.data, use_gpu=not args.cpu, model_name=args.model_name)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["train_regime_model", "main"]
