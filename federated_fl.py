"""Flower-based federated training using the Flower framework."""
from __future__ import annotations

import os
from typing import Optional, Tuple, List

try:  # pragma: no cover - optional dependency
    import flwr as fl
except Exception as exc:  # pragma: no cover - missing dependency
    raise SystemExit(
        "True federated training requires the 'flwr' package."
        " Install it with 'pip install flwr'"
    ) from exc
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from supabase import create_client

from federated_trainer import (
    FederatedEnsemble,
    _load_params,
    _prepare_data,
    _train_client,
)


class _LGBClient(fl.client.NumPyClient):
    """A simple Flower client that trains a LightGBM model."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, params: dict) -> None:
        self.X = X
        self.y = y
        self.params = params
        self.booster: Optional[lgb.Booster] = None

    def get_parameters(self, config: dict) -> List[np.ndarray]:
        if self.booster is None:
            return []
        model_str = self.booster.model_to_string().encode("utf-8")
        arr = np.frombuffer(model_str, dtype=np.uint8)
        return [arr]

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        if parameters:
            model_bytes = bytes(parameters[0])
            self.booster = lgb.Booster(model_str=model_bytes.decode("utf-8"))
        self.booster = _train_client(self.X, self.y, self.params)
        model_str = self.booster.model_to_string().encode("utf-8")
        arr = np.frombuffer(model_str, dtype=np.uint8)
        return [arr], len(self.X), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[float, int, dict]:
        if self.booster is None and parameters:
            model_bytes = bytes(parameters[0])
            self.booster = lgb.Booster(model_str=model_bytes.decode("utf-8"))
        if self.booster is None:
            return 0.0, len(self.X), {}
        preds = self.booster.predict(self.X)
        y_pred = preds.argmax(axis=1) if preds.ndim > 1 else (preds >= 0.5).astype(int)
        label_map = {-1: 0, 0: 1, 1: 2}
        y_enc = self.y.replace(label_map).astype(int)
        acc = accuracy_score(y_enc, y_pred)
        return float(acc), len(self.X), {"accuracy": acc}


class _SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy that records models from clients."""

    def __init__(self) -> None:
        super().__init__()
        self.models: List[bytes] = []

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        for res, _ in results:
            arr = res.parameters.tensors[0]
            self.models.append(bytes(arr))
        return aggregated


def launch(
    start_ts: str,
    end_ts: str,
    *,
    config_path: str = "cfg.yaml",
    table: str = "ohlc_data",
    params_override: Optional[dict] = None,
    num_clients: int = 3,
    num_rounds: int = 1,
) -> Tuple[FederatedEnsemble, dict]:
    """Run a Flower-based federated LightGBM training simulation."""

    params = _load_params(config_path)
    if params_override:
        params.update(params_override)

    X, y = _prepare_data(start_ts, end_ts, table=table)
    indices = np.array_split(np.arange(len(X)), num_clients)
    splits = [(X.iloc[idx], y.iloc[idx]) for idx in indices]

    def client_fn(cid: str) -> _LGBClient:
        idx = int(cid)
        X_part, y_part = splits[idx]
        return _LGBClient(X_part, y_part, params)

    strategy = _SaveModelStrategy()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    models = [lgb.Booster(model_str=m.decode("utf-8")) for m in strategy.models]
    ensemble = FederatedEnsemble(models)

    label_map = {-1: 0, 0: 1, 1: 2}
    y_enc = y.replace(label_map).astype(int)
    preds = ensemble.predict(X)
    y_pred = preds.argmax(axis=1) if preds.ndim > 1 else (preds >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_enc, y_pred)),
        "f1": float(f1_score(y_enc, y_pred, average="macro")),
        "precision_long": float(precision_score(y_enc, y_pred, average="macro")),
        "recall_long": float(recall_score(y_enc, y_pred, average="macro")),
        "n_models": len(models),
    }

    joblib.dump(ensemble, "flower_federated_model.pkl")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if url and key:
        try:
            client = create_client(url, key)
            bucket = client.storage.from_("models")
            with open("flower_federated_model.pkl", "rb") as fh:
                bucket.upload("flower_federated_model.pkl", fh)
        except Exception:
            pass

    return ensemble, metrics


def start_server(
    start_ts: str,
    end_ts: str,
    *,
    config_path: str = "cfg.yaml",
    table: str = "ohlc_data",
    params_override: Optional[dict] = None,
    num_rounds: int = 1,
    server_address: str = "0.0.0.0:8080",
) -> Tuple[FederatedEnsemble, dict]:
    """Start a Flower server and return the aggregated model and metrics."""

    strategy = _SaveModelStrategy()
    fl.server.start_server(
        server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    params = _load_params(config_path)
    if params_override:
        params.update(params_override)

    X, y = _prepare_data(start_ts, end_ts, table=table)

    models = [lgb.Booster(model_str=m.decode("utf-8")) for m in strategy.models]
    ensemble = FederatedEnsemble(models)

    label_map = {-1: 0, 0: 1, 1: 2}
    y_enc = y.replace(label_map).astype(int)
    preds = ensemble.predict(X)
    y_pred = preds.argmax(axis=1) if preds.ndim > 1 else (preds >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_enc, y_pred)),
        "f1": float(f1_score(y_enc, y_pred, average="macro")),
        "precision_long": float(precision_score(y_enc, y_pred, average="macro")),
        "recall_long": float(recall_score(y_enc, y_pred, average="macro")),
        "n_models": len(models),
    }

    joblib.dump(ensemble, "flower_federated_model.pkl")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if url and key:
        try:
            client = create_client(url, key)
            bucket = client.storage.from_("models")
            with open("flower_federated_model.pkl", "rb") as fh:
                bucket.upload("flower_federated_model.pkl", fh)
        except Exception:
            pass

    return ensemble, metrics


def start_client(
    start_ts: str,
    end_ts: str,
    *,
    server_address: str = "0.0.0.0:8080",
    config_path: str = "cfg.yaml",
    table: str = "ohlc_data",
    params_override: Optional[dict] = None,
) -> None:
    """Start a Flower client for federated training."""

    params = _load_params(config_path)
    if params_override:
        params.update(params_override)

    X, y = _prepare_data(start_ts, end_ts, table=table)
    client = _LGBClient(X, y, params)

    fl.client.start_numpy_client(server_address, client)


__all__ = [
    "launch",
    "start_server",
    "start_client",
]

