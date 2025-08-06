from __future__ import annotations

"""Train a minimal fallback model with synthetic data."""

import base64
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _load_data(data: pd.DataFrame | str | None) -> tuple[pd.DataFrame, pd.Series]:
    if data is None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 3))
        y = (X.sum(axis=1) > 0).astype(int)
        df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
        return df, pd.Series(y, name="target")
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
    if "target" in df.columns:
        y = df.pop("target")
    else:
        y = df.iloc[:, -1]
        df = df.iloc[:, :-1]
    return df, y


def train_fallback_model(
    data: pd.DataFrame | str | None = None,
    *,
    use_gpu: bool = False,
) -> str:
    """Train a tiny logistic regression model and return base64 weights."""
    X, y = _load_data(data)
    model = LogisticRegression().fit(X, y)
    payload = base64.b64encode(pickle.dumps(model)).decode("utf-8")
    return payload


def main() -> None:
    """Entry point for menu invocation."""
    weights_b64 = train_fallback_model()
    print(weights_b64)


__all__ = ["train_fallback_model", "main"]

if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
