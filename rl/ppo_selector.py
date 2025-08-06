from __future__ import annotations

"""Train a PPO selector on trade data."""

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch


class CustomTradingEnv(gym.Env):
    """Simple trading environment built from trade data."""

    metadata = {"render_modes": []}

    def __init__(self, trades: pd.DataFrame) -> None:
        super().__init__()
        numeric = trades.select_dtypes("number")
        if numeric.empty:
            raise ValueError("trades must contain numeric columns")

        self._features = numeric.to_numpy(dtype=np.float32)
        self._prices = self._features[:, 0]
        self._n_steps = len(self._features) - 1
        self._step = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._features.shape[1],),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._step = 0
        return self._features[self._step], {}

    def step(self, action: int):
        self._step += 1
        done = self._step >= self._n_steps
        price_diff = self._prices[self._step] - self._prices[self._step - 1]
        if action == 1:  # buy
            reward = price_diff
        elif action == 2:  # sell
            reward = -price_diff
        else:  # hold
            reward = 0.0

        obs = self._features[self._step] if not done else self._features[self._n_steps]
        return obs, float(reward), done, False, {}


MODEL_PATH = "rl_selector.zip"


def train(csv_path: str, use_gpu: bool = True, dest: str | None = None) -> str:
    """Train a PPO selector on trade data and optionally upload it.

    Parameters
    ----------
    csv_path:
        Path to a CSV file containing trade history.
    use_gpu:
        If ``True`` and CUDA is available, train on the GPU.
    dest:
        Optional Supabase storage destination. When provided the saved model
        is uploaded using :func:`utils.upload.upload_to_supabase`.

    Returns
    -------
    str
        Path to the saved model file.
    """
    trades = pd.read_csv(csv_path)

    def _make_env() -> CustomTradingEnv:
        return CustomTradingEnv(trades)

    env = make_vec_env(_make_env)

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        ent_coef=0.01,
        clip_range=lambda f: 0.2 * (1 - f),
    )
    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save(MODEL_PATH)

    if dest is not None:
        from utils.upload import upload_to_supabase

        upload_to_supabase(MODEL_PATH, dest)

    return MODEL_PATH


__all__ = ["CustomTradingEnv", "train"]
