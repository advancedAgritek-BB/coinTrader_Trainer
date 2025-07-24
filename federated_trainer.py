"""Federated LightGBM training utilities."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from supabase import create_client
import yaml

from coinTrader_Trainer.data_loader import fetch_data_range_async
from coinTrader_Trainer.data_loader import fetch_trade_aggregates
from coinTrader_Trainer.feature_engineering import make_features

__all__ = ["train_federated_regime"]


async def _fetch_async(start: str, end: str) -> pd.DataFrame:
    """Fetch trade logs between ``start`` and ``end`` asynchronously."""
    return await fetch_data_range_async("trade_logs", start, end)


def _load_params(cfg_path: str) -> dict:
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    params = cfg.get("regime_lgbm", {})
    # LightGBM expects the parameter ``device_type`` when selecting the
    # computation backend. Always enable GPU by default to mirror the previous
    # behaviour where ``device`` was forced to ``gpu`` regardless of the
    # configuration value.
    params["device_type"] = "gpu"
    return params


def _prepare_data(start_ts: str | pd.Timestamp, end_ts: str | pd.Timestamp,
                  symbols: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    start = start_ts.isoformat() if isinstance(start_ts, pd.Timestamp) else str(start_ts)
    end = end_ts.isoformat() if isinstance(end_ts, pd.Timestamp) else str(end_ts)

    df = asyncio.run(_fetch_async(start, end))

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if symbols is not None and "symbol" in df.columns:
        df = df[df["symbol"].isin(set(symbols))]

    df = make_features(df, use_gpu=True)
    if "target" not in df.columns:
        raise ValueError("Data must contain a 'target' column for training")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


@dataclass
class FederatedEnsemble:
    models: List[lgb.Booster]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)


def _train_client(X: pd.DataFrame, y: pd.Series, params: dict) -> lgb.Booster:
    dataset = lgb.Dataset(X, label=y)
    num_round = params.get("num_boost_round", 100)
    booster = lgb.train(params, dataset, num_boost_round=num_round)
    return booster


def train_federated_regime(
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    *,
    num_clients: int = 3,
    config_path: str = "cfg.yaml",
    params_override: Optional[dict] = None,
) -> Tuple[FederatedEnsemble, dict]:
    """Train LightGBM models across ``num_clients`` and aggregate their predictions."""

    params = _load_params(config_path)
    if params_override:
        params.update(params_override)

    # Optionally fetch aggregated stats before downloading the full dataset
    try:
        fetch_trade_aggregates(
            pd.to_datetime(start_ts), pd.to_datetime(end_ts)
        )
    except Exception:
        pass

    X, y = _prepare_data(start_ts, end_ts)

    indices = np.array_split(np.arange(len(X)), num_clients)
    models: List[lgb.Booster] = []
    for idx in indices:
        booster = _train_client(X.iloc[idx], y.iloc[idx], params)
        models.append(booster)

    ensemble = FederatedEnsemble(models)

    preds = ensemble.predict(X)
    y_pred = (preds >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "precision_long": float(precision_score(y, y_pred)),
        "recall_long": float(recall_score(y, y_pred)),
        "n_models": len(models),
    }

    joblib.dump(ensemble, "federated_model.pkl")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if url and key:
        try:
            client = create_client(url, key)
            bucket = client.storage.from_("models")
            with open("federated_model.pkl", "rb") as fh:
                bucket.upload("federated_model.pkl", fh)
        except Exception:
            pass

    return ensemble, metrics

