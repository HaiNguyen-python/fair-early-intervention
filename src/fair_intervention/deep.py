"""Optional deep RL baselines (DQN, PPO) with stable-baselines3.

Install with ``pip install -e ".[deep]"`` (pulls in torch). Kept out of the default and dev
dependencies so CI stays fast; ``tests/test_policy.py`` skips the smoke test when the package
is missing. On a state space this small, tabular Q-learning is the right tool; these exist to
check that claim, as in the thesis this repo is based on.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .env import EnvConfig, InterventionEnv

Algo = Literal["dqn", "ppo"]


class SB3Policy:
    """Wrap a trained stable-baselines3 model in this repo's ``Policy`` interface."""

    def __init__(self, model: object, name: str) -> None:
        self.model = model
        self.name = name

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)  # type: ignore[attr-defined]
        return int(action)


def train_sb3(algo: Algo, config: EnvConfig, timesteps: int, seed: int = 0) -> SB3Policy:
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as err:
        raise ImportError('Deep RL needs the optional extra: pip install -e ".[deep]"') from err
    cls = {"dqn": DQN, "ppo": PPO}[algo]
    model = cls("MlpPolicy", InterventionEnv(config), seed=seed, verbose=0)
    model.learn(total_timesteps=timesteps)
    return SB3Policy(model, algo.upper())
