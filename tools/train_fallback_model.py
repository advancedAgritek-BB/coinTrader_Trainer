"""Wrapper to train the fallback model and upload via ModelRegistry."""

from __future__ import annotations

from typing import Optional

from registry import ModelRegistry
from train_fallback_model import train_fallback_model as _train_fallback_model


def main(
    data: str | None = None,
    *,
    use_gpu: bool = False,
    model_name: str = "fallback_model",
    registry: Optional[ModelRegistry] = None,
):
    """Train the fallback model and upload its weights to Supabase."""
    payload = _train_fallback_model(data, use_gpu=use_gpu)
    if registry is None:
        registry = ModelRegistry()
    registry.upload_dict({"weights_b64": payload}, model_name)
    return payload


__all__ = ["main"]
from __future__ import annotations

"""Train a tiny LightGBM fallback model and store it as base64."""

import base64
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb


def main() -> None:
    """Generate synthetic data, train model, and write base64 to file."""
    # Synthetic dataset
    X = np.random.rand(100, 5)
    y = np.random.choice([0, 1, 2], 100)

    params = dict(n_estimators=20, num_leaves=10, device="gpu")

    try:
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
    except Exception:
        params["device"] = "cpu"
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)

    payload = base64.b64encode(pickle.dumps(model)).decode("utf-8")
    out_path = Path(__file__).resolve().parent.parent / "fallback_b64.txt"
    out_path.write_text(payload)


if __name__ == "__main__":  # pragma: no cover
    main()
