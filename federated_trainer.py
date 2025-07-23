"""Federated training utilities for regime model."""

from __future__ import annotations

import asyncio
import os
from typing import Iterable, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from lightgbm import Booster
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_loader import fetch_data_range_async
from feature_engineering import make_features
from registry import ModelRegistry


async def _async_fetch(start_ts: str, end_ts: str) -> pd.DataFrame:
    """Internal helper to fetch trade logs asynchronously."""
    df = await fetch_data_range_async("trade_logs", start_ts, end_ts)
    return df


def fetch_and_prepare_data(
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    symbols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch trade data and return feature matrix ``X`` and targets ``y``."""

    if isinstance(start_ts, pd.Timestamp):
        start_ts = start_ts.isoformat()
    if isinstance(end_ts, pd.Timestamp):
        end_ts = end_ts.isoformat()

    df = asyncio.run(_async_fetch(str(start_ts), str(end_ts)))

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})

    if symbols is not None and "symbol" in df.columns:
        symbols_set = set(symbols)
        df = df[df["symbol"].isin(symbols_set)]

    df = make_features(df)

    if "target" not in df.columns:
        raise ValueError("Data must contain a 'target' column for training")

    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def client_train(
    X_local: pd.DataFrame,
    y_local: pd.Series,
    params: dict,
) -> Booster:
    """Train a LightGBM model on ``X_local`` and ``y_local``."""

    train_set = lgb.Dataset(X_local, label=y_local)
    num_round = params.get("num_boost_round", 100)
    booster = lgb.train(params, train_set, num_boost_round=num_round)
    return booster


def federated_aggregate(models: Iterable[Booster]) -> Callable[[pd.DataFrame], np.ndarray]:
    """Return an ensemble prediction function averaging model outputs."""

    model_list = list(models)

    def predict(X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in model_list])
        return preds.mean(axis=1)

    return predict


def _load_params(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("regime_lgbm", {})


def train_federated_regime(
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    num_clients: int = 3,
    config_path: str = "cfg.yaml",
) -> Tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    """Train models on federated splits and return aggregated predictor."""

    params = _load_params(config_path)

    X, y = fetch_and_prepare_data(start_ts, end_ts)

    indices = np.array_split(np.arange(len(X)), num_clients)
    models: List[Booster] = []
    for idx in indices:
        X_local = X.iloc[idx]
        y_local = y.iloc[idx]
        booster = client_train(X_local, y_local, params)
        models.append(booster)

    ensemble = federated_aggregate(models)

    preds = ensemble(X)
    y_pred = (preds >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "precision_long": float(precision_score(y, y_pred)),
        "recall_long": float(recall_score(y, y_pred)),
    }

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if url and key:
        try:
            registry = ModelRegistry(url, key)
            registry.upload(ensemble, "federated_regime", metrics)
        except Exception:
            pass

    return ensemble, metrics

