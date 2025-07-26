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

load_dotenv()


def _train_single(df: pd.DataFrame, params: dict) -> lgb.Booster:
    X = df.drop(columns=["target"], errors="ignore")
    y = df.get("target")
    dataset = lgb.Dataset(X, label=y)
    booster = lgb.train(params, dataset)
    return booster


def train_federated_regime(
    X: pd.DataFrame | None,
    y: pd.Series | None,
    params: dict,
    use_gpu: bool = True,
    *,
    n_clients: int = 1,
    registry: Optional[ModelRegistry] = None,
    model_name: str = "federated_regime",
) -> Tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    """Train multiple LightGBM models and return an ensemble callable."""

    models: List[lgb.Booster] = []
    for _ in range(n_clients):
        df = asyncio.run(fetch_data_range_async("ohlc_data", "start", "end"))
        df = make_features(df)
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
