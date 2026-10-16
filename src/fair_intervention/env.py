"""A small Gymnasium simulator of one term of early interventions with a shared budget.

One episode is one term for a cohort. Each week the agent sees the enrolled students one at a
time and decides: no action, an email nudge (cheap) or a tutor call (expensive). After the
decision the student either stays or drops out this week.

Observation (all discrete, so tabular Q-learning works):
    risk bucket (0-4) | week | budget bin | group (0 other, 1 low-IMD) | previous contacts (0-2+)

Objective of one term (what the school cares about):
    retained - cost_weight * spend - lambda * final coverage gap
The coverage gap is |share of truly at-risk low-IMD students contacted at least once - the same
share for other students|.

Reward per step uses potential-based shaping (Ng, Harada and Russell, 1999). The potential is
the expected number of students who finish the term, given everyone's current dropout hazard.
A step's reward is the change in that potential (the acted-on student's survival chance before
the decision vs after the decision and this week's outcome), minus the cost, minus lambda times
the change in the coverage gap. Over an episode the rewards add up to
    objective - (expected survivors at the start, a constant the policy cannot change),
so the best policy is unchanged, but the agent sees the long-term effect of a call at the step
where it made the call instead of weeks later on another student's turn.

Planted to match part 1: some students carry a hidden extra risk that the risk bucket does not
show at first, and this is more common in the low-IMD group. From ``reveal_week`` the bucket
catches up. A policy that trusts the early bucket alone under-serves that group.

**Every effect size here is an assumption** (see docs/decisions.md). The simulator is a tool to
compare policies under stated assumptions, not evidence that an email reduces dropout by 15%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

NO_ACTION, EMAIL, CALL = 0, 1, 2
ActionName = Literal["none", "email", "call"]
ACTION_NAMES: tuple[ActionName, ...] = ("none", "email", "call")
N_BUCKETS = 5
MAX_CONTACTS = 2
# Weekly dropout probability by observed risk bucket (about 8% to 75% over an 8-week term).
BUCKET_HAZARD = (0.01, 0.025, 0.05, 0.09, 0.16)
BUCKET_SHARES = (0.30, 0.25, 0.20, 0.15, 0.10)


@dataclass(frozen=True)
class EnvConfig:
    n_students: int = 40
    n_weeks: int = 8
    budget: float = 40.0
    email_cost: float = 1.0
    call_cost: float = 5.0
    # How many retained students one unit of budget is worth giving up: 0.02 means a tutor call
    # (5 units) must save at least 0.1 expected dropouts to pay for itself.
    cost_weight: float = 0.02
    email_effect: float = 0.2
    call_effect: float = 0.6
    # Each repeat contact with the same student works half as well as the previous one.
    effect_decay: float = 0.5
    low_group_share: float = 0.4
    hidden_risk_low: float = 0.3
    hidden_risk_other: float = 0.08
    hidden_multiplier: float = 3.0
    # From this week on the hidden risk shows up in the data: the observed bucket jumps by two.
    # Waiting buys information, so spending the whole budget in week 0 is not automatically best.
    reveal_week: int = 3
    # A student counts as "truly at risk" for the coverage gap above this weekly hazard.
    at_risk_hazard: float = 0.05
    fairness_lambda: float = 0.0
    n_budget_bins: int = 4


def budget_bin(budget_left: float, config: EnvConfig) -> int:
    """0 = cannot afford a call; higher bins = larger share of the budget left."""
    if budget_left < config.call_cost:
        return 0
    frac = budget_left / config.budget
    return 1 + min(config.n_budget_bins - 2, int(frac * (config.n_budget_bins - 1)))


class InterventionEnv(gym.Env[np.ndarray, np.int64]):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        c = self.config
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete(
            [N_BUCKETS, c.n_weeks, c.n_budget_bins, 2, MAX_CONTACTS + 1]
        )
        self._costs = np.array([0.0, c.email_cost, c.call_cost])
        self._effects = np.array([0.0, c.email_effect, c.call_effect])

    # -- episode setup ---------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        c, rng = self.config, self.np_random
        n = c.n_students
        self.bucket = rng.choice(N_BUCKETS, size=n, p=BUCKET_SHARES)
        self.group = (rng.random(n) < c.low_group_share).astype(int)
        hidden = rng.random(n) < np.where(self.group == 1, c.hidden_risk_low, c.hidden_risk_other)
        self.hidden = hidden
        self.hazard = np.array(BUCKET_HAZARD)[self.bucket] * np.where(
            hidden, c.hidden_multiplier, 1.0
        )
        self.hazard = np.minimum(self.hazard, 0.9)
        self.truly_at_risk = self.hazard >= c.at_risk_hazard
        # Counts per group, updated in place: the gap is needed twice per step.
        self.at_risk_n = np.bincount(self.group[self.truly_at_risk], minlength=2).tolist()
        self.covered_n = [0, 0]
        self.retained = n
        # Pre-drawn dropout noise: the same seed gives the same cohort and the same luck for
        # every policy (common random numbers), which makes policy comparisons less noisy.
        self.noise = rng.random((n, c.n_weeks))
        self.enrolled = np.ones(n, dtype=bool)
        self.contacts = np.zeros(n, dtype=int)
        self.covered = np.zeros(n, dtype=bool)
        self.budget_left = float(c.budget)
        self.spent = 0.0
        self.week = 0
        self.queue = list(range(n))
        self.current = self.queue.pop(0)
        return self._obs(), self._info()

    # -- observation helpers ---------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        i = self.current
        bucket = int(self.bucket[i])
        if self.hidden[i] and self.week >= self.config.reveal_week:
            bucket = min(N_BUCKETS - 1, bucket + 2)
        return np.array(
            (
                bucket,
                self.week,
                budget_bin(self.budget_left, self.config),
                self.group[i],
                min(self.contacts[i], MAX_CONTACTS),
            ),
            dtype=np.int64,
        )

    def coverage_gap(self) -> float:
        n0, n1 = self.at_risk_n
        if n0 == 0 or n1 == 0:
            return 0.0
        return abs(self.covered_n[1] / n1 - self.covered_n[0] / n0)

    def _info(self) -> dict[str, Any]:
        return {
            "retained": self.retained,
            "spent": self.spent,
            "budget_left": self.budget_left,
            "coverage_gap": self.coverage_gap(),
        }

    # -- dynamics --------------------------------------------------------------------------
    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        c, i = self.config, self.current
        action = int(action)
        if self._costs[action] > self.budget_left:
            action = NO_ACTION  # cannot spend money the school does not have
        cost = float(self._costs[action])
        gap_before = self.coverage_gap()
        survival_before = (1.0 - self.hazard[i]) ** (c.n_weeks - self.week)
        if action != NO_ACTION:
            effect = self._effects[action] * c.effect_decay ** self.contacts[i]
            self.hazard[i] *= 1.0 - effect
            if self.truly_at_risk[i] and not self.covered[i]:
                self.covered_n[self.group[i]] += 1
            self.contacts[i] += 1
            self.covered[i] = True
            self.budget_left -= cost
            self.spent += cost
        dropped = bool(self.noise[i, self.week] < self.hazard[i])
        weeks_after = c.n_weeks - self.week - 1
        survival_after = 0.0 if dropped else (1.0 - self.hazard[i]) ** weeks_after
        if dropped:
            self.enrolled[i] = False
            self.retained -= 1
        reward = (
            survival_after
            - survival_before
            - c.cost_weight * cost
            - c.fairness_lambda * (self.coverage_gap() - gap_before)
        )
        terminated = self._advance()
        return self._obs(), reward, terminated, False, self._info()

    def _advance(self) -> bool:
        """Move to the next enrolled student; start a new week when the queue is empty."""
        while not self.queue:
            self.week += 1
            if self.week >= self.config.n_weeks:
                self.week = self.config.n_weeks - 1  # keep the final observation in range
                return True
            self.queue = np.flatnonzero(self.enrolled).tolist()
            if not self.queue:
                return True
        self.current = self.queue.pop(0)
        return False
