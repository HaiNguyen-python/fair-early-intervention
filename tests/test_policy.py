"""The simulator's contract (determinism, budget, rewards) and the learning agents."""

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from fair_intervention.agents import QLearningAgent, RandomPolicy, ThresholdPolicy
from fair_intervention.env import CALL, EMAIL, NO_ACTION, EnvConfig, InterventionEnv
from fair_intervention.policy_eval import (
    episode_runs,
    evaluate_policy,
    lambda_sweep,
    mean_ci,
    paired_difference,
)

SMALL = EnvConfig(n_students=20, n_weeks=6, budget=25.0)


def _trajectory(seed: int, actions: list[int]):
    env = InterventionEnv(SMALL)
    obs, _ = env.reset(seed=seed)
    out = [obs.tolist()]
    for a in actions:
        obs, r, done, _, info = env.step(a)
        out.append((obs.tolist(), r, done, info["retained"]))
        if done:
            break
    return out


def test_env_passes_the_gymnasium_checker():
    check_env(InterventionEnv(SMALL), skip_render_check=True)


def test_env_is_deterministic_for_a_seed():
    actions = list(np.random.default_rng(0).integers(0, 3, 200))
    assert _trajectory(5, actions) == _trajectory(5, actions)
    assert _trajectory(5, actions) != _trajectory(6, actions)


def test_budget_never_goes_negative_and_unaffordable_actions_are_free():
    env = InterventionEnv(SMALL)
    for seed in range(20):
        env.reset(seed=seed)
        done = False
        while not done:
            spent_before = env.spent
            _, _, done, _, info = env.step(CALL)
            assert info["budget_left"] >= 0
            assert info["spent"] - spent_before in (0.0, SMALL.call_cost)
        assert info["spent"] <= SMALL.budget


def test_rewards_add_up_to_the_objective_minus_a_policy_free_constant():
    """Potential-based shaping: sum of rewards = retained - cost - lambda*gap - start potential."""
    cfg = replace(SMALL, fairness_lambda=3.0)
    env = InterventionEnv(cfg)
    for policy in (ThresholdPolicy(), RandomPolicy(seed=1)):
        obs, _ = env.reset(seed=9)
        start_potential = float(np.sum((1 - env.hazard) ** cfg.n_weeks))
        total, done = 0.0, False
        while not done:
            obs, r, done, _, info = env.step(policy.act(obs))
            total += r
        objective = info["retained"] - cfg.cost_weight * info["spent"] - 3.0 * info["coverage_gap"]
        assert total == pytest.approx(objective - start_potential)


def test_a_call_lowers_hazard_with_diminishing_effect():
    env = InterventionEnv(SMALL)
    env.reset(seed=2)
    i, h0 = env.current, env.hazard[env.current]
    env.step(CALL)
    first = 1 - env.hazard[i] / h0
    env.current, h1 = i, env.hazard[i]
    env.enrolled[i] = True
    env.step(CALL)
    second = 1 - env.hazard[i] / h1
    assert first == pytest.approx(SMALL.call_effect)
    assert second == pytest.approx(SMALL.call_effect * SMALL.effect_decay)


def test_threshold_rule_calls_top_risk_once():
    rule = ThresholdPolicy()
    assert rule.act(np.array([4, 0, 3, 0, 0])) == CALL
    assert rule.act(np.array([3, 0, 3, 1, 0])) == EMAIL
    assert rule.act(np.array([4, 2, 3, 0, 1])) == NO_ACTION
    assert rule.act(np.array([1, 0, 3, 0, 0])) == NO_ACTION


def test_q_learning_beats_random_on_the_same_cohorts():
    agent = QLearningAgent(SMALL)
    agent.train(600, seed=0)
    q = episode_runs(agent, SMALL, 150)
    rnd = episode_runs(RandomPolicy(seed=3), SMALL, 150)
    diff, half_width = paired_difference(q, rnd, "objective")
    assert diff - half_width > 0  # better, and the 95% interval excludes zero


def test_fairness_weight_shrinks_the_coverage_gap():
    rows = lambda_sweep(SMALL, [0.0, 20.0], train_episodes=600, eval_episodes=100)
    assert rows[1]["coverage_gap"][0] < rows[0]["coverage_gap"][0]


def test_q_table_round_trip_and_shape_check(tmp_path):
    agent = QLearningAgent(SMALL)
    agent.q[0, 0, 0, 0, 0, CALL] = 1.0
    agent.save(tmp_path / "q.npy")
    loaded = QLearningAgent.load(tmp_path / "q.npy", SMALL)
    assert loaded.act(np.array([0, 0, 0, 0, 0])) == CALL
    with pytest.raises(ValueError, match="shape"):
        QLearningAgent.load(tmp_path / "q.npy", EnvConfig())


def test_mean_ci_and_evaluation_shape():
    assert mean_ci([2.0]) == (2.0, 0.0)
    mean, hw = mean_ci([1.0, 3.0])
    assert mean == 2.0 and hw == pytest.approx(1.96 * np.sqrt(2) / np.sqrt(2))
    stats = evaluate_policy(ThresholdPolicy(), SMALL, 5)
    assert set(stats) == {"retained", "spent", "coverage_gap", "objective"}


def test_optional_deep_baseline_runs_when_installed():
    pytest.importorskip("stable_baselines3")
    from fair_intervention.deep import train_sb3

    policy = train_sb3("ppo", SMALL, timesteps=2_000)
    assert policy.act(np.array([4, 0, 3, 0, 0])) in (NO_ACTION, EMAIL, CALL)
