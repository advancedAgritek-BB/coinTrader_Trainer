from __future__ import annotations

"""Utilities for training reinforcement-learning agents using PPO."""

from typing import Any, Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from registry import ModelRegistry


def train_ppo(
    data: Any,
    use_gpu: bool = False,
    total_timesteps: int = 10000,
    model_path: str = "ppo_model.zip",
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
    model = PPO("MlpPolicy", env, verbose=1, device=device)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model


def train(
    data: Any,
    *,
    use_gpu: bool = False,
    total_timesteps: int = 10000,
    model_name: str = "ppo_selector",
    registry: ModelRegistry | None = None,
) -> PPO:
    """Train a PPO selector and upload it via :class:`ModelRegistry`."""

    model = train_ppo(data, use_gpu=use_gpu, total_timesteps=total_timesteps)
    if registry is None:
        registry = ModelRegistry()
    registry.upload(model, model_name)
    return model


__all__ = ["train_ppo", "train"]

