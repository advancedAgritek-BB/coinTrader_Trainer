from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

import data_loader

@dataclass
class Agent:
    params: Dict[str, Any]
    fitness: float = float("nan")

async def run_swarm_simulation(
    num_agents: int = 3,
    start_ts: str | None = None,
    end_ts: str | None = None,
) -> Tuple[Dict[str, Any], List[Agent]]:
    """Run a minimal swarm simulation returning best parameters.

    Parameters are purely illustrative and this function serves as a
    placeholder for real optimisation logic. It trains a LightGBM model
    for each agent using data fetched via ``fetch_data_range_async``.
    """

    df = await data_loader.fetch_data_range_async(
        "trade_logs", start_ts or "start", end_ts or "end"
    )
    if "target" in df.columns:
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        X = df
        y = pd.Series(np.zeros(len(df)))

    agents: List[Agent] = [Agent({"agent_id": i}) for i in range(num_agents)]
    for agent in agents:
        lgb.train(agent.params, lgb.Dataset(X, label=y), num_boost_round=1)
        agent.fitness = 1.0

    best_agent = agents[0]
    return best_agent.params, agents
from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import networkx as nx
import lightgbm as lgb

from data_loader import fetch_data_range_async
from feature_engineering import make_features


@dataclass
class SwarmAgent:
    """Lightweight agent holding LightGBM parameters and fitness."""

    id: int
    params: Dict[str, Any]
    fitness: float = field(default=float("inf"))

    def simulate(self, X: pd.DataFrame, y: pd.Series, base_params: Dict[str, Any]) -> None:
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
        dataset = lgb.Dataset(X, label=y)
        booster = lgb.train(
            train_params,
            dataset,
            num_boost_round=train_params.get("num_boost_round", 10),
            verbose_eval=False,
        )
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
            new_params[key] = float(np.mean(vals))
        agent.params = new_params


async def run_swarm_simulation(
    start_ts: datetime, end_ts: datetime, num_agents: int = 50
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
    df = await fetch_data_range_async(
        "trade_logs", start_ts.isoformat(), end_ts.isoformat()
    )
    df = make_features(df)
    if "target" not in df.columns:
        df["target"] = (df["price"].shift(-1) > df["price"]).astype(int).fillna(0)
    X = df.drop(columns=["target"])
    y = df["target"]

    with open("cfg.yaml", "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    base_params: Dict[str, Any] = cfg.get("regime_lgbm", {})

    graph = nx.Graph()
    agents: List[SwarmAgent] = []
    rng = np.random.default_rng(0)
    for i in range(num_agents):
        params = dict(base_params)
        lr = params.get("learning_rate", 0.1) * float(rng.uniform(0.5, 1.5))
        params["learning_rate"] = lr
        agents.append(SwarmAgent(i, params))
        graph.add_node(i, agent=agents[-1])
        if i > 0:
            graph.add_edge(i - 1, i)

    rounds = 3
    for _ in range(rounds):
        for agent in agents:
            agent.simulate(X, y, base_params)
        evolve_swarm(agents, graph)

    best = min(agents, key=lambda a: a.fitness)
    return best.params

