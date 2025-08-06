from __future__ import annotations

"""Utilities for training reinforcement-learning agents."""

from typing import Any

from stable_baselines3 import PPO


def train_ppo(env: Any, total_timesteps: int = 1000, *, use_gpu: bool = False, **kwargs) -> PPO:
    """Train a PPO agent on ``env``.

    Parameters
    ----------
    env : Any
        Environment compatible with Stable Baselines 3.
    total_timesteps : int, optional
        Number of timesteps to train for.
    use_gpu : bool, optional
        When ``True`` the model is initialised with
        ``policy_kwargs={'device': 'cuda'}``; otherwise ``'cpu'`` is used.
    kwargs : dict
        Additional keyword arguments passed to :class:`stable_baselines3.PPO`.
    """

    device = "cuda" if use_gpu else "cpu"
    policy_kwargs = kwargs.pop("policy_kwargs", {})
    policy_kwargs.setdefault("device", device)
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model
