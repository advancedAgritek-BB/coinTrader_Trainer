from __future__ import annotations

import asyncio
import os
from typing import Callable, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_loader import fetch_data_range_async
from feature_engineering import make_features
from registry import ModelRegistry
from utils import timed

load_dotenv()


def _train_single(df: pd.DataFrame, params: dict) -> lgb.Booster:
    X = df.drop(columns=["target"], errors="ignore")
    y = df.get("target")
    dataset = lgb.Dataset(X, label=y)
    booster = lgb.train(params, dataset)
    return booster


@timed
def train_federated_regime(
    start_ts: pd.Timestamp | str | pd.DataFrame,
    end_ts: pd.Timestamp | str | None,
    params: dict,
    use_gpu: bool = True,
    *,
    n_clients: int = 1,
    table: str = "ohlc_data",
    symbol: Optional[str] = None,
    registry: Optional[ModelRegistry] = None,
    model_name: str = "federated_regime",
) -> Tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    """Train multiple LightGBM models and return an ensemble callable.

    Parameters
    ----------
    start_ts : pd.Timestamp | str | pd.DataFrame
        Data start timestamp, or a preloaded DataFrame containing the training
        data. If a DataFrame is provided, ``end_ts`` is ignored.
    end_ts : pd.Timestamp | str | None
        End of the training period. Must be provided when ``start_ts`` is not a
        DataFrame.
    params : dict
        LightGBM training parameters.
    use_gpu : bool
        Whether to enable GPU feature generation.
    n_clients : int, optional
        Number of federated clients to simulate.
    table : str, optional
        Supabase table to fetch data from.
    symbol : str, optional
        Trading pair symbol to filter fetched data.
    registry : ModelRegistry, optional
        Registry instance to upload the resulting models to.
    model_name : str, optional
        Name under which to upload the models.
    """

    if isinstance(start_ts, pd.DataFrame):
        source_df = start_ts
    else:
        if end_ts is None:
            raise ValueError("end_ts must be provided when start_ts is not a DataFrame")
        start = start_ts.isoformat() if hasattr(start_ts, "isoformat") else str(start_ts)
        end = end_ts.isoformat() if hasattr(end_ts, "isoformat") else str(end_ts)
        source_df = asyncio.run(fetch_data_range_async(table, start, end))
        if symbol is not None and "symbol" in source_df.columns:
            source_df = source_df[source_df["symbol"] == symbol]

    models: List[lgb.Booster] = []
    for _ in range(n_clients):
        df = make_features(source_df.copy(), use_gpu=use_gpu)
        booster = _train_single(df, params)
        models.append(booster)

    def ensemble(data: pd.DataFrame) -> np.ndarray:
        preds = [m.predict(data) for m in models]
        return np.mean(preds, axis=0)

    metrics = {"n_models": len(models)}

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if url and key:
        try:
            env_reg = ModelRegistry(url, key)
            env_reg.upload(models, model_name, metrics)
        except Exception:
            pass
    if registry is not None:
        try:
            registry.upload(models, model_name, metrics)
        except Exception:
            pass
    return ensemble, metrics
