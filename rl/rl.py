from __future__ import annotations

"""Utilities for training reinforcement-learning agents using PPO."""

from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from cointrainer.registry import ModelRegistry


def train_ppo(
    data: Any,
    use_gpu: bool = False,
    total_timesteps: int = 10000,
    model_path: str = "ppo_model.zip",
    *,
    learning_rate: float = 3e-4,
    exploration: float = 0.0,
) -> PPO:
    """Train a Proximal Policy Optimisation (PPO) agent.

    Parameters
    ----------
    data : Any
        A gymnasium-compatible environment instance or a callable returning one.
    use_gpu : bool, optional
        If ``True`` and CUDA is available, training occurs on the GPU; otherwise
        the CPU is used.
    total_timesteps : int, optional
        Number of timesteps to train for.
    model_path : str, optional
        Destination path where the trained model will be saved. A ``.zip``
        extension is recommended and automatically appended by
        :mod:`stable_baselines3`.

    Returns
    -------
    PPO
        The trained model instance.
    """

    # Build a vectorized environment from ``data`` when necessary
    if callable(data):
        env = DummyVecEnv([data])
    else:
        env = data
        if not hasattr(env, "num_envs"):
            env = DummyVecEnv([lambda: data])

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=learning_rate,
        ent_coef=exploration,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model


def train(
    data: Any,
    *,
    use_gpu: bool = False,
    total_timesteps: int = 10000,
    model_name: str = "ppo_selector",
    learning_rate: float = 3e-4,
    exploration: float = 0.0,
    registry: ModelRegistry | None = None,
) -> PPO:
    """Train a PPO selector and upload it via :class:`ModelRegistry`."""

    model_path = f"{model_name}.zip"
    model = train_ppo(
        data,
        use_gpu=use_gpu,
        total_timesteps=total_timesteps,
        model_path=model_path,
        learning_rate=learning_rate,
        exploration=exploration,
    )
    if registry is None:
        registry = ModelRegistry()
    registry.upload_file(model_path, model_name)
    return model


__all__ = ["train", "train_ppo"]

