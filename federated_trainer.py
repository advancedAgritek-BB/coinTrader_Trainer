"""Federated LightGBM training utilities."""
# isort: skip_file

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from supabase import SupabaseException, create_client
import httpx

from coinTrader_Trainer.data_loader import (
    fetch_data_range_async,
    fetch_trade_aggregates,
)
from coinTrader_Trainer.feature_engineering import make_features

load_dotenv()


__all__ = ["train_federated_regime"]


async def _fetch_async(
    start: str, end: str, *, table: str = "ohlc_data"
) -> pd.DataFrame:
    """Fetch trade logs between ``start`` and ``end`` asynchronously."""
    return await fetch_data_range_async(table, start, end)


def _load_params(cfg_path: str) -> dict:
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    params = cfg.get("regime_lgbm", {})
    # LightGBM expects the parameter ``device_type`` when selecting the
    # computation backend. Always enable GPU by default to mirror the previous
    # behaviour where ``device`` was forced to ``gpu`` regardless of the
    # configuration value.
    params["device_type"] = "gpu"
    params.setdefault("objective", "multiclass")
    params.setdefault("num_class", 3)
    return params


def _prepare_data(
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    symbols: Optional[Iterable[str]] = None,
    *,
    table: str = "ohlc_data",
    min_rows: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and targets between ``start_ts`` and ``end_ts``.

    Parameters
    ----------
    min_rows : int, optional
        Minimum number of rows required. A ``ValueError`` is raised when fewer
        rows are returned.
    """
    start = (
        start_ts.isoformat() if isinstance(start_ts, pd.Timestamp) else str(start_ts)
    )
    end = end_ts.isoformat() if isinstance(end_ts, pd.Timestamp) else str(end_ts)

    df = asyncio.run(_fetch_async(start, end, table=table))
    if df.empty:
        logging.error("No data returned for %s - %s", start, end)
        raise ValueError("No data available")

    if df.empty or len(df) < min_rows:
        raise ValueError(
            f"Expected at least {min_rows} rows of data, got {len(df)}"
        )

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
        preds = [m.predict(X) for m in self.models]
        preds = [p if p.ndim == 2 else p.reshape(len(p), -1) for p in preds]
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0)


def _train_client(X: pd.DataFrame, y: pd.Series, params: dict) -> lgb.Booster:
    label_map = {-1: 0, 0: 1, 1: 2}
    y_enc = y.replace(label_map).astype(int)
    dataset = lgb.Dataset(X, label=y_enc)
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
    table: str = "ohlc_data",
) -> Tuple[FederatedEnsemble, dict]:
    """Train LightGBM models across ``num_clients`` and aggregate their predictions."""

    params = _load_params(config_path)
    if params_override:
        params.update(params_override)

    # Optionally fetch aggregated stats before downloading the full dataset
    try:
        fetch_trade_aggregates(pd.to_datetime(start_ts), pd.to_datetime(end_ts))
    except (httpx.HTTPError, SupabaseException, ValueError, TypeError, AttributeError) as exc:  # pragma: no cover
        logging.exception("Failed to fetch aggregates: %s", exc)

    X, y = _prepare_data(start_ts, end_ts, table=table)
    label_map = {-1: 0, 0: 1, 1: 2}
    y_enc = y.replace(label_map).astype(int)

    indices = np.array_split(np.arange(len(X)), num_clients)
    models: List[lgb.Booster] = []
    for idx in indices:
        booster = _train_client(X.iloc[idx], y.iloc[idx], params)
        models.append(booster)

    ensemble = FederatedEnsemble(models)

    preds = ensemble.predict(X)
    if preds.ndim > 1:
        y_pred = np.argmax(preds, axis=1)
    else:
        y_pred = (preds >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_enc, y_pred)),
        "f1": float(f1_score(y_enc, y_pred, average="macro")),
        "precision_long": float(precision_score(y_enc, y_pred, average="macro")),
        "recall_long": float(recall_score(y_enc, y_pred, average="macro")),
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
        except (httpx.HTTPError, SupabaseException) as exc:  # pragma: no cover
            logging.exception("Failed to upload model: %s", exc)

    return ensemble, metrics
