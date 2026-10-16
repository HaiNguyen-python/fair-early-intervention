"""Evaluate intervention policies over many seeded episodes, with 95% confidence intervals.

Evaluation seeds start at 100_000, far from the training seeds (0, 1, 2, ...), so a policy is
never scored on a cohort it trained on. Every policy sees the same cohorts and the same dropout
noise (see ``InterventionEnv.reset``), so differences between policies are not luck of the draw.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from .agents import NeverPolicy, Policy, QLearningAgent, RandomPolicy, ThresholdPolicy
from .env import EnvConfig, InterventionEnv

EVAL_SEED = 100_000


def mean_ci(values: Sequence[float]) -> tuple[float, float]:
    """Mean and 95% half-width (normal approximation; fine for 100+ episodes)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def run_episode(env: InterventionEnv, policy: Policy, seed: int) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)
    done = False
    info: dict[str, float] = {}
    while not done:
        obs, _, done, _, info = env.step(policy.act(obs))
    c = env.config
    return {
        "retained": info["retained"],
        "spent": info["spent"],
        "coverage_gap": info["coverage_gap"],
        "objective": info["retained"]
        - c.cost_weight * info["spent"]
        - c.fairness_lambda * info["coverage_gap"],
    }


def episode_runs(
    policy: Policy, config: EnvConfig, episodes: int, seed: int = EVAL_SEED
) -> list[dict[str, float]]:
    env = InterventionEnv(config)
    return [run_episode(env, policy, seed + i) for i in range(episodes)]


def summarise(runs: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    return {k: mean_ci([r[k] for r in runs]) for k in runs[0]}


def evaluate_policy(
    policy: Policy, config: EnvConfig, episodes: int, seed: int = EVAL_SEED
) -> dict[str, tuple[float, float]]:
    return summarise(episode_runs(policy, config, episodes, seed))


def paired_difference(
    a: list[dict[str, float]], b: list[dict[str, float]], key: str
) -> tuple[float, float]:
    """Mean and 95% half-width of a - b over the same seeded episodes.

    Both policies faced the same cohorts and the same dropout noise, so the per-episode
    difference cancels most of the cohort-to-cohort variation. This interval is much narrower
    than comparing two separate means, and it is the honest one to quote for "A beats B".
    """
    return mean_ci([x[key] - y[key] for x, y in zip(a, b, strict=True)])


def baselines() -> list[Policy]:
    return [NeverPolicy(), RandomPolicy(seed=7), ThresholdPolicy()]


def lambda_sweep(
    config: EnvConfig,
    lambdas: Sequence[float],
    train_episodes: int,
    eval_episodes: int,
    seed: int = 0,
    on_trained: Callable[[float, QLearningAgent], None] | None = None,
) -> list[dict[str, object]]:
    """Train one Q-learning agent per fairness weight and evaluate it.

    All results are measured with the *same* evaluation config except lambda, so retention,
    spend and coverage gap are directly comparable across rows.
    """
    rows: list[dict[str, object]] = []
    for lam in lambdas:
        cfg = replace(config, fairness_lambda=lam)
        agent = QLearningAgent(cfg)
        agent.train(train_episodes, seed=seed)
        if on_trained is not None:
            on_trained(lam, agent)
        rows.append({"lambda": lam, **evaluate_policy(agent, cfg, eval_episodes)})
    return rows
