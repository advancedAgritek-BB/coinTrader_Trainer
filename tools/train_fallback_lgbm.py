from __future__ import annotations

"""Train a tiny LightGBM fallback model and store it as base64."""

import base64
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np


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


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
