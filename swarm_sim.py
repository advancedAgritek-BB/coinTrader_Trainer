from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
import asyncio

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from utils import timed, validate_schema
import httpx
from supabase import SupabaseException

import data_loader
from feature_engineering import make_features
from registry import ModelRegistry
from train_pipeline import check_clinfo_gpu
from train_pipeline import check_clinfo_gpu, verify_lightgbm_gpu
from sklearn.utils import resample

load_dotenv()


async def fetch_and_prepare_data(
    start_ts: datetime | str,
    end_ts: datetime | str,
    *,
    table: str = "ohlc_data",
    return_threshold: float = 0.01,
    min_rows: int = 1,
    generate_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch trade data and return feature matrix ``X`` and targets ``y``.

    Parameters
    ----------
    return_threshold : float, optional
        Threshold used to generate the ``target`` column when it is missing.
    min_rows : int, optional
        Minimum number of rows required. A ``ValueError`` is raised when fewer
        rows are returned.
    """

    if isinstance(start_ts, datetime):
        start_ts = start_ts.isoformat()
    if isinstance(end_ts, datetime):
        end_ts = end_ts.isoformat()

    df = await data_loader.fetch_data_range_async(table, str(start_ts), str(end_ts))
    if df.empty:
        logging.error("No data returned for %s - %s", start_ts, end_ts)
        raise ValueError("No data available")

    if df.empty or len(df) < min_rows:
        raise ValueError(
            f"Expected at least {min_rows} rows of data, got {len(df)}"
        )

    if "timestamp" in df.columns and "ts" not in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
    if "ts" not in df.columns:
        try:
            start_dt = pd.to_datetime(start_ts)
        except (TypeError, ValueError):
            start_dt = pd.Timestamp.utcnow()
        df["ts"] = pd.date_range(start_dt, periods=len(df), freq="min")
    validate_schema(df, ["ts"])

    loop = asyncio.get_running_loop()
    try:
        df = await loop.run_in_executor(
            None,
            lambda: make_features(df, generate_target=generate_target),
        )
    except TypeError:
        df = await loop.run_in_executor(None, lambda: make_features(df))
    except ValueError:
        pass

    if "target" not in df.columns:
        returns = df["price"].pct_change().shift(-1)
        df["target"] = pd.Series(
            np.where(
                returns > return_threshold,
                1,
                np.where(returns < -return_threshold, -1, 0),
            ),
            index=df.index,
        ).fillna(0)

    try:
        counts = df["target"].value_counts()
        max_count = counts.max()
        if len(counts) > 1 and max_count > 0:
            frames = [
                resample(g, replace=True, n_samples=max_count, random_state=42)
                for _, g in df.groupby("target")
            ]
            df = (
                pd.concat(frames)
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )
    except Exception:
        logging.exception("Failed to balance labels")

    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


@dataclass
class SwarmAgent:
    """Lightweight agent holding LightGBM parameters and fitness."""

    id: int
    params: Dict[str, Any]
    fitness: float = field(default=float("inf"))

    async def simulate(
        self, X: pd.DataFrame, y: pd.Series, base_params: Dict[str, Any]
    ) -> None:
        """Train a small LightGBM model and update ``fitness``.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target labels.
        base_params : Dict[str, Any]
            Baseline LightGBM parameters to start from.
        """
        train_params = {**base_params, **self.params}
        train_params.setdefault("device_type", "gpu")
        dataset = lgb.Dataset(X, label=y)
        loop = asyncio.get_running_loop()
        try:
            booster = await loop.run_in_executor(
                None,
                lambda: lgb.train(
                    train_params,
                    dataset,
                    num_boost_round=train_params.get("num_boost_round", 10),
                ),
            )
        except lgb.basic.LightGBMError as exc:
            if "OpenCL" in str(exc):
                logging.exception("LightGBM GPU training failed: %s", exc)
                train_params["device_type"] = "cpu"
                booster = await loop.run_in_executor(
                    None,
                    lambda: lgb.train(
                        train_params,
                        dataset,
                        num_boost_round=train_params.get("num_boost_round", 10),
                    ),
                )
            else:
                raise
        preds = booster.predict(X)
        error = np.mean((preds - y) ** 2)
        self.fitness = float(error)


def evolve_swarm(agents: List[SwarmAgent], graph: nx.Graph) -> None:
    """Average each agent's parameters with its neighbors.

    Parameters
    ----------
    agents : List[SwarmAgent]
        List of all agents in the swarm.
    graph : nx.Graph
        Graph describing neighborhood relations between agents.
    """
    for agent in agents:
        neighbor_ids = list(graph.neighbors(agent.id))
        if not neighbor_ids:
            continue
        neighbors = [graph.nodes[n]["agent"] for n in neighbor_ids]
        all_params = [agent.params] + [n.params for n in neighbors]
        new_params: Dict[str, Any] = {}
        for key in agent.params.keys():
            vals = [p.get(key, agent.params[key]) for p in all_params]
            if all(isinstance(v, (int, float, np.number)) for v in vals):
                new_params[key] = float(np.mean(vals))
            else:
                new_params[key] = vals[0]
        agent.params = new_params


@timed
async def run_swarm_search(
    start_ts: datetime,
    end_ts: datetime,
    num_agents: int = 50,
    *,
    table: str = "ohlc_data",
) -> Dict[str, Any]:
    """Run an asynchronous swarm optimisation simulation.

    Parameters
    ----------
    start_ts : datetime
        Inclusive start timestamp for the training data.
    end_ts : datetime
        Exclusive end timestamp for the training data.
    num_agents : int, optional
        Number of agents in the swarm. Defaults to ``50``.

    Returns
    -------
    Dict[str, Any]
        Parameter dictionary from the best-performing agent.
    """
    X, y = await fetch_and_prepare_data(start_ts, end_ts, table=table)

    with open("cfg.yaml", "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    base_params: Dict[str, Any] = cfg.get("regime_lgbm", {})
    if check_clinfo_gpu():
        base_params.setdefault("device_type", "gpu")

    if check_clinfo_gpu() and verify_lightgbm_gpu(base_params):
        base_params.setdefault("device_type", "gpu")
        base_params.setdefault("gpu_platform_id", 0)
        base_params.setdefault("gpu_device_id", 0)
    else:
        base_params["device_type"] = "cpu"
        logging.warning("GPU not detected; falling back to CPU")

    graph = nx.complete_graph(num_agents)
    agents: List[SwarmAgent] = []
    rng = np.random.default_rng(0)
    for i in range(num_agents):
        params = dict(base_params)
        lr = params.get("learning_rate", 0.1) * float(rng.uniform(0.5, 1.5))
        params["learning_rate"] = lr
        agents.append(SwarmAgent(i, params))
        graph.nodes[i]["agent"] = agents[-1]

    rounds = 3
    for _ in range(rounds):
        await asyncio.gather(
            *(agent.simulate(X, y, base_params) for agent in agents)
        )
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: evolve_swarm(agents, graph)
        )

    best = min(agents, key=lambda a: a.fitness)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    bucket = os.environ.get("PARAMS_BUCKET", "agent_params")
    table = os.environ.get("PARAMS_TABLE", "agent_params")
    if url and key:
        try:
            reg = ModelRegistry(url, key, bucket=bucket, table=table)
            entry_id = reg.upload_dict(
                best.params,
                "swarm_params",
                {"fitness": best.fitness},
                conflict_key="name",
            )
            logging.info("Uploaded swarm parameters %s", entry_id)
        except (httpx.HTTPError, SupabaseException) as exc:  # pragma: no cover
            logging.exception("Failed to upload parameters: %s", exc)
    else:
        logging.info("SUPABASE credentials not set; skipping parameter upload")

    return best.params
