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

__all__ = [
    "LightGBMClient",
    "booster_to_parameters",
    "parameters_to_booster",
    "run_federated_fl",
]
