"""Flower-based federated training using the Flower framework."""
from __future__ import annotations

import os
from typing import Optional, Tuple, List

import flwr as fl
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

from __future__ import annotations

"""Flower-based federated LightGBM training."""

from dataclasses import dataclass
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from datasets import Dataset

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr_datasets.partitioner import IidPartitioner

from coinTrader_Trainer.federated_trainer import _load_params, _prepare_data, _train_client


@dataclass
class LightGBMClient(fl.client.NumPyClient):
    """Flower client wrapping a LightGBM booster."""

    X: pd.DataFrame
    y: pd.Series
    params: dict
    model: lgb.Booster | None = None

    def get_parameters(self, config: dict | None = None):  # type: ignore[override]
        if self.model is None:
            booster = _train_client(self.X, self.y, self.params)
            self.model = booster
        return booster_to_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore[override]
        if parameters and parameters.tensors:
            self.model = parameters_to_booster(parameters)
        self.model = _train_client(self.X, self.y, self.params)
        return booster_to_parameters(self.model), len(self.y), {}

    def evaluate(self, parameters, config):  # type: ignore[override]
        if parameters and parameters.tensors:
            self.model = parameters_to_booster(parameters)
        preds = self.model.predict(self.X)  # type: ignore[union-attr]
        label_map = {-1: 0, 0: 1, 1: 2}
        y_true = self.y.replace(label_map).astype(int)
        if preds.ndim > 1:
            y_pred = preds.argmax(axis=1)
        else:
            y_pred = (preds >= 0.5).astype(int)
        accuracy = float(np.mean(y_pred == y_true))
        return 0.0, len(self.y), {"accuracy": accuracy}


def booster_to_parameters(booster: lgb.Booster) -> fl.common.Parameters:
    """Serialize a LightGBM booster to Flower parameters."""
    raw = booster.model_to_string().encode()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return ndarrays_to_parameters([arr])


def parameters_to_booster(params: fl.common.Parameters) -> lgb.Booster:
    """Deserialize Flower parameters back into a LightGBM booster."""
    arr = parameters_to_ndarrays(params)[0]
    model_str = arr.tobytes().decode()
    return lgb.Booster(model_str=model_str)


def _make_client_fn(X: pd.DataFrame, y: pd.Series, params: dict):
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    dataset = Dataset.from_pandas(df)
    partitioner = IidPartitioner(num_partitions=10)
    partitioner.dataset = dataset

    def client_fn(cid: str) -> fl.client.Client:
        part = partitioner.load_partition(int(cid)).to_pandas()
        X_part = part.drop(columns=["target"])
        y_part = part["target"]
        return LightGBMClient(X_part, y_part, params)

    return client_fn


def run_federated_fl(start_ts, end_ts, *, config_path="cfg.yaml", table="ohlc_data"):
    """Run Flower simulation using LightGBM clients."""
    params = _load_params(config_path)
    X, y = _prepare_data(start_ts, end_ts, table=table)
    client_fn = _make_client_fn(X, y, params)

    strategy = FedAvg(min_available_clients=10, min_fit_clients=10)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=ServerConfig(num_rounds=20),
        strategy=strategy,
    )


def start_server(*, address: str = "0.0.0.0:8080", num_rounds: int = 1) -> None:
    """Launch a Flower aggregation server."""
    fl.server.start_server(
        server_address=address,
        config=ServerConfig(num_rounds=num_rounds),
    )


def start_client(
    start_ts: str,
    end_ts: str,
    *,
    address: str = "127.0.0.1:8080",
    config_path: str = "cfg.yaml",
    table: str = "ohlc_data",
) -> None:
    """Connect to a Flower server and train on local data."""
    params = _load_params(config_path)
    X, y = _prepare_data(start_ts, end_ts, table=table)
    client = _LGBClient(X, y, params)
    fl.client.start_numpy_client(address, client)

__all__ = [
    "LightGBMClient",
    "booster_to_parameters",
    "parameters_to_booster",
    "start_server",
    "start_client",
    "run_federated_fl",
]
