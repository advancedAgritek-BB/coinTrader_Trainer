from __future__ import annotations

"""Train a regime classification model using LightGBM.

This script loads a dataset of trades containing a ``volatility`` column,
constructs ordinal regime labels and performs a parameter grid search for a
LightGBM classifier. The best model is saved to ``regime_lgbm.pkl`` and can be
optionally uploaded to Supabase.
"""


import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from utils.upload import upload_to_supabase


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def load_trades(path: str) -> pd.DataFrame:
    """Return a DataFrame loaded from ``path``.

    ``path`` may reference a CSV or Parquet file. The file is expected to
    contain a ``volatility`` column which is used to derive the regime labels.
    """

    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(path)
    if file.suffix == ".csv":
        return pd.read_csv(file)
    if file.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file)
    raise ValueError(f"Unsupported file type: {file.suffix}")


def build_labels(df: pd.DataFrame, low: float | None, med: float | None) -> Tuple[pd.DataFrame, float, float]:
    """Derive ordinal regime labels from the ``volatility`` column.

    Parameters
    ----------
    df:
        Data containing a ``volatility`` column.
    low, med:
        Thresholds for low and medium volatility. When omitted they default to
        the 33rd and 66th percentiles respectively.
    """

    if "volatility" not in df.columns:
        raise ValueError("Input data must include a 'volatility' column")

    vol = df["volatility"].astype(float)
    if low is None or med is None:
        q = vol.quantile([0.33, 0.66]).to_list()
        low = low if low is not None else q[0]
        med = med if med is not None else q[1]

    df["target"] = pd.cut(vol, bins=[0, low, med, np.inf], labels=[0, 1, 2]).astype(int)
    return df, low, med


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split ``df`` into features and target labels."""

    y = df.pop("target").astype(int)
    # Drop volatility column and keep only numeric features
    X = df.drop(columns=["volatility"]).select_dtypes("number").copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(X: pd.DataFrame, y: pd.Series, *, use_gpu: bool = False) -> Tuple[lgb.LGBMClassifier, float, float]:
    """Run GridSearchCV to fit a LightGBM model.

    Returns the best estimator along with F1-macro and ROC-AUC scores computed
    on a held out validation set.
    """

    estimator = lgb.LGBMClassifier(
        device="gpu" if use_gpu else "cpu",
        objective="multiclass",
        num_class=3,
    )

    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "n_estimators": [100, 200, 300],
    }

    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    grid.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(10)],
    )

    y_pred = grid.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, grid.predict_proba(X_test), multi_class="ovr")

    return grid.best_estimator_, f1, roc_auc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train a regime LightGBM model")
    parser.add_argument("data", help="Path to trades file with a volatility column")
    parser.add_argument("--low", type=float, default=None, help="Low volatility threshold")
    parser.add_argument("--med", type=float, default=None, help="Medium volatility threshold")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU training")
    parser.add_argument("--output", default="regime_lgbm.pkl", help="Model output path")
    parser.add_argument(
        "--upload", metavar="DEST", default=None, help="Optional Supabase upload destination"
    )
    args = parser.parse_args()

    df = load_trades(args.data)
    df, low, med = build_labels(df, args.low, args.med)
    X, y = prepare_xy(df)
    model, f1, roc_auc = train_model(X, y, use_gpu=args.use_gpu)

    with open(args.output, "wb") as fh:
        pickle.dump(model, fh)

    print(f"F1-macro: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    if args.upload:
        upload_to_supabase(args.output, args.upload)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
