"""Federated LightGBM training utilities."""
# isort: skip_file

from __future__ import annotations

import asyncio
import concurrent.futures
import multiprocessing
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from cointrainer.features.build import make_features
from utils import timed, validate_schema
from supabase import SupabaseException, create_client
import httpx

from cointrainer.data.loader import fetch_data_range_async
from cointrainer.data.cache import get_cache

load_dotenv()


__all__ = ["train_federated_regime"]

# Mapping from regime label to LightGBM class index
LABEL_MAP = {-1: 0, 0: 1, 1: 2}


async def _fetch_async(
    start: str, end: str, *, table: str = "ohlc_data"
) -> pd.DataFrame:
    """Fetch trade logs between ``start`` and ``end`` asynchronously."""
    return await fetch_data_range_async(table, start, end)


def _load_params(cfg_path: str) -> dict:
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    params = cfg.get("regime_lgbm", {})
    # LightGBM expects the ``device`` parameter when selecting the
    # computation backend. Always enable GPU by default to mirror the previous
    # behaviour where ``device`` was forced to ``gpu`` regardless of the
    # configuration value.
    params["device"] = "opencl"
    params.setdefault("objective", "multiclass")
    params.setdefault("num_class", 3)
    return params


async def _prepare_data(
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    symbols: Optional[Iterable[str]] = None,
    *,
    table: str = "ohlc_data",
    min_rows: int = 1,
    redis_client: Any | None = None,
    cache_key: str | None = None,
    generate_target: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and targets between ``start_ts`` and ``end_ts``.

    Parameters
    ----------
    min_rows : int, optional
        Minimum number of rows required. A ``ValueError`` is raised when fewer
        rows are returned.
    generate_target : bool, optional
        Whether to create the ``target`` column when it is missing.
    """
    start = (
        start_ts.isoformat() if isinstance(start_ts, pd.Timestamp) else str(start_ts)
    )
    end = end_ts.isoformat() if isinstance(end_ts, pd.Timestamp) else str(end_ts)

    df = await _fetch_async(start, end, table=table)
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
    validate_schema(df, ["ts"])

    try:
        df = make_features(
            df,
            use_gpu=True,
            redis_client=redis_client,
            cache_key=cache_key,
            generate_target=generate_target,
        )
    except TypeError:
        df = make_features(
            df,
            use_gpu=True,
            redis_client=redis_client,
            cache_key=cache_key,
        )
    if "target" not in df.columns:
        raise ValueError("Data must contain a 'target' column for training")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def prepare_data(
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    symbols: Optional[Iterable[str]] = None,
    *,
    table: str = "ohlc_data",
    min_rows: int = 1,
    redis_client: Any | None = None,
    cache_key: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Synchronous wrapper around ``_prepare_data`` for convenience."""

    return asyncio.run(
        _prepare_data(
            start_ts,
            end_ts,
            symbols,
            table=table,
            min_rows=min_rows,
            redis_client=redis_client,
            cache_key=cache_key,
        )
    )


@dataclass
class FederatedEnsemble:
    models: List[lgb.Booster]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]
        preds = [p if p.ndim == 2 else p.reshape(len(p), -1) for p in preds]
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0)


def _train_client(X: pd.DataFrame, y: pd.Series, params: dict) -> lgb.Booster:
    y_enc = y.replace(LABEL_MAP).astype(int)
    dataset = lgb.Dataset(X, label=y_enc)
    train_params = dict(params)
    num_round = train_params.get("num_boost_round", 100)
    try:
        booster = lgb.train(train_params, dataset, num_boost_round=num_round)
    except lgb.basic.LightGBMError as exc:  # pragma: no cover - hardware dependent
        if "OpenCL" in str(exc):
            logging.exception("LightGBM GPU training failed: %s", exc)
            train_params["device"] = "cpu"
            train_params.pop("device_type", None)
            booster = lgb.train(train_params, dataset, num_boost_round=num_round)
        else:
            raise
    return booster


@timed
async def train_federated_regime(
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    *,
    num_clients: int = 3,
    config_path: str = "cfg.yaml",
    params_override: Optional[dict] = None,
    table: str = "ohlc_data",
    feature_cache_key: str | None = None,
    use_processes: bool = True,
) -> Tuple[FederatedEnsemble, dict]:
    """Train LightGBM models across ``num_clients`` and aggregate their predictions.

    Parameters
    ----------
    use_processes : bool
        When ``True`` (default), train each client in a separate process. Set to
        ``False`` to run the clients in threads instead.
    """

    params = _load_params(config_path)
    if params_override:
        params.update(params_override)

    redis_client = get_cache() if feature_cache_key else None
    X, y = await _prepare_data(
        start_ts,
        end_ts,
        table=table,
        redis_client=redis_client,
        cache_key=feature_cache_key,
        generate_target=True,
    )
    y_enc = y.replace(LABEL_MAP).astype(int)

    indices = np.array_split(np.arange(len(X)), num_clients)
    loop = asyncio.get_running_loop()
    if use_processes:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_clients, mp_context=ctx
        ) as executor:
            tasks = [
                loop.run_in_executor(
                    executor, _train_client, X.iloc[idx], y.iloc[idx], params
                )
                for idx in indices
            ]
            models: List[lgb.Booster] = list(await asyncio.gather(*tasks))
    else:
        tasks = [
            asyncio.to_thread(_train_client, X.iloc[idx], y.iloc[idx], params)
            for idx in indices
        ]
        models: List[lgb.Booster] = list(await asyncio.gather(*tasks))

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
            raise

    return ensemble, metrics
