from __future__ import annotations

"""Simple Q-learning bot using swarm simulation for experience generation.

This module provides :class:`RLBot`, a lightweight reinforcement learning
agent that learns Q-values for a discrete set of trading strategies.  Training
samples are obtained by running :func:`swarm_sim.run_swarm_search`, which acts as
an environment simulator providing parameter dictionaries.  Numeric parameters
are interpreted as state vectors and a synthetic reward is derived from them.

The trained bot can select among provided strategy names using an epsilon-greedy
policy.  Model persistence leverages :class:`cointrainer.registry.ModelRegistry`
mirroring the behaviour of other RL utilities in this repository.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

import asyncio
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from cointrainer.registry import ModelRegistry
from swarm_sim import run_swarm_search


class QNetwork(nn.Module):
    """Minimal fully-connected network mapping states to Q-values."""

    def __init__(self, state_dim: int, n_actions: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple forward
        return self.layers(x)


@dataclass
class RLBot:
    """Q-learning agent choosing between trading strategies."""

    strategies: Sequence[str]
    state_dim: int
    lr: float = 1e-3
    gamma: float = 0.99
    use_gpu: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_dim, len(self.strategies)).to(self.device)
        self.optimizer = Adam(self.q_net.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Experience generation -------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _params_to_tensor(params: dict[str, float]) -> torch.Tensor:
        """Convert swarm parameter dictionary to a state tensor."""
        numeric = [float(v) for v in params.values() if isinstance(v, (int, float, np.number))]
        return torch.tensor(numeric, dtype=torch.float32)

    @staticmethod
    def _reward_from_params(params: dict[str, float]) -> float:
        """Derive a synthetic reward from swarm parameters."""
        numeric = [float(v) for v in params.values() if isinstance(v, (int, float, np.number))]
        return -float(np.mean(numeric)) if numeric else 0.0

    def generate_experience(self, start_ts: datetime, end_ts: datetime, table: str = "ohlc_data") -> tuple[torch.Tensor, float]:
        """Run swarm search and return a state tensor and reward."""
        params = asyncio.run(run_swarm_search(start_ts, end_ts, table=table, num_agents=5))
        state = self._params_to_tensor(params)
        reward = self._reward_from_params(params)
        return state.to(self.device), reward

    # ------------------------------------------------------------------
    # Q-learning ---------------------------------------------------------
    # ------------------------------------------------------------------
    def select_action(self, state: torch.Tensor, epsilon: float = 0.1) -> str:
        """Return a strategy name using an epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            idx = int(np.random.randint(len(self.strategies)))
        else:
            with torch.no_grad():
                q_values = self.q_net(state.unsqueeze(0))
                idx = int(torch.argmax(q_values).item())
        return self.strategies[idx]

    def _select_index(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """Internal helper returning the action index."""
        if np.random.rand() < epsilon:
            return int(np.random.randint(len(self.strategies)))
        with torch.no_grad():
            q_values = self.q_net(state.unsqueeze(0))
            return int(torch.argmax(q_values).item())

    def update(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor) -> None:
        """Perform a Q-learning update step."""
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        q_values = self.q_net(state)[0, action]
        with torch.no_grad():
            next_q = self.q_net(next_state).max(1)[0]
            target = reward + self.gamma * next_q
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ----------------------------------------------------------------------
# Training utility -----------------------------------------------------
# ----------------------------------------------------------------------

def train(
    *,
    episodes: int = 10,
    strategies: Iterable[str] | None = None,
    use_gpu: bool = False,
    model_name: str = "rl_bot",
    registry: ModelRegistry | None = None,
) -> RLBot:
    """Train :class:`RLBot` and upload it via :class:`ModelRegistry`.

    Parameters
    ----------
    episodes:
        Number of swarm episodes to generate.
    strategies:
        Sequence of strategy names the bot can select.
    use_gpu:
        Whether to run the Q-network on the GPU if available.
    model_name:
        Identifier used when persisting the model.
    registry:
        Optional :class:`ModelRegistry` instance for uploading.
    """

    strategies = list(strategies) if strategies else ["sniper_solana", "lstm_bot"]
    start = datetime.utcnow() - timedelta(days=1)
    end = datetime.utcnow()
    init_params = asyncio.run(run_swarm_search(start, end, num_agents=5))
    state_dim = len([v for v in init_params.values() if isinstance(v, (int, float, np.number))]) or 1
    bot = RLBot(strategies=strategies, state_dim=state_dim, use_gpu=use_gpu)

    state = bot._params_to_tensor(init_params).to(bot.device)
    for _ in range(episodes):
        action_idx = bot._select_index(state)
        next_state, reward = bot.generate_experience(start, end)
        bot.update(state, action_idx, reward, next_state)
        state = next_state

    if registry is None:
        registry = ModelRegistry()
    registry.upload(bot, model_name)
    return bot


__all__ = ["QNetwork", "RLBot", "train"]
