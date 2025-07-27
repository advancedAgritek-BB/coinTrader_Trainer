from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from data_loader import fetch_data_range_async
from feature_engineering import make_features
from utils import validate_schema


DEFAULT_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "min_data_in_leaf": 50,
    "num_boost_round": 200,
    "objective": "regression",
    "metric": "mse",
    "verbose": -1,
}


async def load_data(
    start_ts: datetime | str,
    end_ts: datetime | str,
    table: str = "ohlc_data",
    *,
    generate_target: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch data between ``start_ts`` and ``end_ts`` and return ``(X, y)``."""

    if isinstance(start_ts, datetime):
        start_ts = start_ts.isoformat()
    if isinstance(end_ts, datetime):
        end_ts = end_ts.isoformat()

    df = await fetch_data_range_async(table, str(start_ts), str(end_ts))
    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    validate_schema(df, ["ts"])
    loop = asyncio.get_running_loop()
    try:
        df = await loop.run_in_executor(
            None, lambda: make_features(df, generate_target=generate_target)
        )
    except TypeError:
        df = await loop.run_in_executor(None, lambda: make_features(df))
    if "target" not in df.columns:
        df["target"] = df["price"].shift(-1).fillna(df["price"]).astype(float)

    X = df.drop(columns=["target"])
    y = df["target"].astype(float)
    return X, y


def _build_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1)
    params["num_leaves"] = trial.suggest_int("num_leaves", 31, 127)
    params["max_depth"] = trial.suggest_int("max_depth", 5, 15)
    params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.6, 1.0)
    params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.6, 1.0)
    params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 20, 100)
    params["num_boost_round"] = trial.suggest_int("num_boost_round", 100, 400)
    return params


def objective_factory(X: pd.DataFrame, y: pd.Series):
    def objective(trial: optuna.Trial) -> float:
        params = _build_params(trial)
        split = int(len(X) * 0.8)
        X_train, X_valid = X.iloc[:split], X.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]
        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_valid, label=y_valid)
        booster = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=params.get("num_boost_round", 100),
            verbose_eval=False,
        )
        preds = booster.predict(X_valid)
        mse = float(np.mean((preds - y_valid) ** 2))
        return mse

    return objective


async def run_optuna_search(
    start_ts: datetime | str,
    end_ts: datetime | str,
    *,
    table: str = "ohlc_data",
    n_trials: int = 100,
    direction: str = "minimize",
) -> Dict[str, Any]:
    """Run an Optuna hyperparameter search and return the best parameters."""

    X, y = await load_data(start_ts, end_ts, table)
    study = optuna.create_study(direction=direction)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: study.optimize(objective_factory(X, y), n_trials=n_trials)
    )
    return study.best_params
