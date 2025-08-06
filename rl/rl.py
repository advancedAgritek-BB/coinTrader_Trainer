"""Reinforcement learning utilities using PPO from stable_baselines3."""

from typing import Any, Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


def train_ppo(
    data: Any,
    use_gpu: bool = False,
    total_timesteps: int = 10000,
    model_path: str = "ppo_model.zip",
):
    """Train a PPO agent on the provided environment.

    Parameters
    ----------
    data: Any
        A gymnasium-compatible environment instance or a callable returning one.
    use_gpu: bool, optional
        If True and CUDA is available, training occurs on the GPU.
    total_timesteps: int, optional
        Number of timesteps to train for.
    model_path: str, optional
        Path where the trained model will be saved. ``.zip`` extension is
        recommended and added automatically by Stable Baselines3.

    Returns
    -------
    PPO
        The trained model instance.
    """

    # Build vectorized environment from the data/environment
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
