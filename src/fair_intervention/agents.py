"""Intervention policies: tabular Q-learning and three baselines.

Every policy maps an observation ``[bucket, week, budget_bin, group, contacts]`` to an action.
The baselines answer "is learning worth it?": never intervening shows the cost of doing
nothing, random shows what spending without a plan buys, and the threshold rule is what a
school would do without any model of the budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .env import CALL, EMAIL, MAX_CONTACTS, N_BUCKETS, NO_ACTION, EnvConfig, InterventionEnv


class Policy(Protocol):
    name: str

    def act(self, obs: np.ndarray) -> int: ...


@dataclass
class NeverPolicy:
    name: str = "Never intervene"

    def act(self, obs: np.ndarray) -> int:
        return NO_ACTION


@dataclass
class RandomPolicy:
    seed: int = 0
    name: str = "Random"
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(3))


@dataclass
class ThresholdPolicy:
    """Call each top-bucket student once, email each second-highest-bucket student once."""

    name: str = "Threshold rule"

    def act(self, obs: np.ndarray) -> int:
        bucket, contacts = int(obs[0]), int(obs[4])
        if contacts > 0:
            return NO_ACTION
        if bucket >= N_BUCKETS - 1:
            return CALL
        if bucket == N_BUCKETS - 2:
            return EMAIL
        return NO_ACTION


@dataclass
class QLearningAgent:
    """Tabular Q-learning with an epsilon-greedy schedule.

    The state space is tiny (5 x weeks x budget bins x 2 x 3 = 960 states for the default
    config), so a table is enough and every learned value can be inspected.
    """

    config: EnvConfig
    alpha: float = 0.05
    gamma: float = 0.995
    name: str = "Q-learning"
    q: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        c = self.config
        shape = (N_BUCKETS, c.n_weeks, c.n_budget_bins, 2, MAX_CONTACTS + 1, 3)
        self.q = np.zeros(shape)

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.q[tuple(obs)]))

    def train(
        self,
        episodes: int,
        seed: int = 0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
    ) -> list[float]:
        """Train on episodes seeded ``seed, seed+1, ...``; returns the return of each episode."""
        env = InterventionEnv(self.config)
        rng = np.random.default_rng(seed)
        returns = []
        for ep in range(episodes):
            # Linear decay over the first 70% of episodes, then mostly greedy.
            eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (0.7 * episodes))
            obs, _ = env.reset(seed=seed + ep)
            done, total = False, 0.0
            while not done:
                s = tuple(obs)
                a = int(rng.integers(3)) if rng.random() < eps else int(np.argmax(self.q[s]))
                obs, r, done, _, _ = env.step(a)
                target = r if done else r + self.gamma * self.q[tuple(obs)].max()
                self.q[s + (a,)] += self.alpha * (target - self.q[s + (a,)])
                total += r
            returns.append(total)
        return returns

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q)

    @classmethod
    def load(cls, path: Path, config: EnvConfig | None = None) -> QLearningAgent:
        agent = cls(config or EnvConfig())
        q = np.load(path)
        if q.shape != agent.q.shape:
            raise ValueError(f"Q-table shape {q.shape} does not match config {agent.q.shape}")
        agent.q = q
        return agent
