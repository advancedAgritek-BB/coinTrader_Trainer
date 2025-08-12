"""Contextual bandit strategy selector using simple linear models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from cointrainer.registry import ModelRegistry


@dataclass
class ContextualBanditStrategySelector:
    """Epsilon-greedy contextual bandit with linear reward models.

    Parameters
    ----------
    n_actions: int
        Number of strategies (arms) available.
    context_dim: int
        Dimensionality of the context vector.
    epsilon: float, optional
        Exploration rate for epsilon-greedy policy.
    lr: float, optional
        Learning rate for the underlying models.
    use_gpu: bool, optional
        If True and CUDA is available, train models on the GPU.
    """

    n_actions: int
    context_dim: int
    epsilon: float = 0.1
    lr: float = 0.01
    use_gpu: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.models = [nn.Linear(self.context_dim, 1).to(self.device) for _ in range(self.n_actions)]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=self.lr) for m in self.models]

    def select(self, context: Sequence[float]) -> int:
        """Select an action according to the current policy."""
        context_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        rewards = [m(context_t).item() for m in self.models]
        return int(np.argmax(rewards))

    def update(self, context: Sequence[float], action: int, reward: float) -> None:
        """Update the model for the chosen action."""
        context_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        model = self.models[action]
        opt = self.optimizers[action]
        opt.zero_grad()
        pred = model(context_t).squeeze()
        loss = (pred - reward_t).pow(2).mean()
        loss.backward()
        opt.step()

    def train(self, data, context_cols: Iterable[str], action_col: str, reward_col: str) -> None:
        """Train the selector on a DataFrame of logged bandit interactions."""
        for _, row in data.iterrows():
            context = row[list(context_cols)].values.astype(float)
            action = int(row[action_col])
            reward = float(row[reward_col])
            self.update(context, action, reward)

    def save(self, path: str) -> None:
        """Persist the selector models to ``path`` using :func:`torch.save`."""
        state = {
            "n_actions": self.n_actions,
            "context_dim": self.context_dim,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "model_state_dicts": [m.state_dict() for m in self.models],
        }
        torch.save(state, path)


def train(
    data: str,
    *,
    context_cols: Iterable[str] | None = None,
    action_col: str = "strategy",
    reward_col: str = "pnl",
    use_gpu: bool = False,
    model_name: str = "bandit_selector",
    registry: ModelRegistry | None = None,
) -> ContextualBanditStrategySelector:
    """Train a contextual bandit selector from logged interactions."""
    df = pd.read_csv(data)
    context_cols = list(context_cols) if context_cols else [
        c for c in df.columns if c not in {action_col, reward_col}
    ]
    selector = ContextualBanditStrategySelector(
        n_actions=int(df[action_col].nunique()),
        context_dim=len(context_cols),
        use_gpu=use_gpu,
    )
    selector.train(df, context_cols, action_col, reward_col)
    if registry is None:
        registry = ModelRegistry()
    registry.upload(selector, model_name)
    return selector


__all__ = ["ContextualBanditStrategySelector", "train"]
