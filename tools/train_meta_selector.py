from __future__ import annotations

"""Train a LightGBM meta selector from trade history."""

from pathlib import Path
from typing import Tuple

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder


def _load_trades(csv_path: str | Path) -> pd.DataFrame:
    """Load trades from ``csv_path``.

    The expected columns are ``strategy``, ``side``, ``entry_time``,
    ``exit_time``, ``entry_price``, ``exit_price`` and ``pnl``. Additional
    columns are ignored.
    """
    df = pd.read_csv(csv_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"])  # type: ignore[assignment]
    df["exit_time"] = pd.to_datetime(df["exit_time"])  # type: ignore[assignment]
    df["duration"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds()
    return df


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct a feature matrix ``X`` and target vector ``y`` from trades."""
    features = ["strategy", "side", "entry_price", "exit_price", "duration"]
    X = df[features].copy()
    y = df["pnl"].copy()

    encoder = OrdinalEncoder()
    X[["strategy", "side"]] = encoder.fit_transform(X[["strategy", "side"]])
    return X, y


def train_meta_selector(trades_csv: str | Path, *, use_gpu: bool = False) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor to predict trade PnL.

    Parameters
    ----------
    trades_csv:
        Path to a CSV file containing trade history.
    use_gpu:
        When ``True`` LightGBM will run on the GPU.
    """
    df = _load_trades(trades_csv)
    X, y = _build_features(df)

    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63],
        "n_estimators": [50, 100, 200],
    }

    device = "gpu" if use_gpu else "cpu"
    estimator = lgb.LGBMRegressor(device=device, early_stopping_rounds=10)
    grid = GridSearchCV(
        estimator,
        param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )

    grid.fit(X, y, eval_set=[(X, y)])
    best_model: lgb.LGBMRegressor = grid.best_estimator_

    preds = best_model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"MSE: {mse:.6f}")
    print(f"R2: {r2:.6f}")

    model_path = Path("meta_selector_lgbm.pkl")
    joblib.dump(best_model, model_path)

    # Optionally upload the model to Supabase
    # from utils.upload import upload_to_supabase
    # upload_to_supabase(str(model_path), f"models/{model_path.name}")

    return best_model


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Train a LightGBM meta selector")
    parser.add_argument("trades_csv", help="Path to trades CSV file")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU training",
    )
    args = parser.parse_args()
    train_meta_selector(args.trades_csv, use_gpu=args.use_gpu)
