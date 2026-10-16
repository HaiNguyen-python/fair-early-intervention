"""FastAPI service: student features in, calibrated risk, reasons and a recommended action out.

The risk comes from the week's calibrated logistic model. The recommended action comes from the
trained Q-learning policy when ``artifacts/q_table.npy`` exists, otherwise from the threshold
rule, so the API still works before any RL training. Group membership (IMD band) is used only
by the intervention policy, never by the risk model.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .agents import Policy, QLearningAgent, ThresholdPolicy
from .data.synthetic import LOW_IMD_BANDS
from .env import ACTION_NAMES, MAX_CONTACTS, EnvConfig, budget_bin
from .features import HORIZONS
from .risk import ModelBundle

# Calibrated risk -> the simulator's five buckets. An assumption that links the two parts;
# it is documented in docs/decisions.md and should be re-checked against real outcomes.
RISK_BUCKET_EDGES = (0.2, 0.4, 0.6, 0.8)

app = FastAPI(title="Fair early intervention", version=__version__)


def artifacts_dir() -> Path:
    return Path(os.getenv("FAIR_INTERVENTION_ARTIFACTS", "artifacts"))


class StudentFeatures(BaseModel):
    clicks_total: float = Field(..., ge=0, description="VLE clicks since registration")
    clicks_last_week: float = Field(..., ge=0)
    clicks_trend: float = Field(..., description="log(1+last week) - log(1+week before)")
    active_days: float = Field(..., ge=0)
    distinct_sites: float = Field(..., ge=0)
    days_since_last_click: float = Field(..., ge=0)
    clicks_before_start: float = Field(..., ge=0)
    n_submitted: float = Field(..., ge=0)
    n_missed: float = Field(..., ge=0)
    n_late: float = Field(..., ge=0)
    mean_score: float | None = Field(None, ge=0, le=100, description="None if nothing marked")
    num_of_prev_attempts: int = Field(..., ge=0)
    studied_credits: int = Field(..., gt=0)


class ScoreRequest(BaseModel):
    week: Literal[4, 6, 8] = Field(..., description="Early-warning week the features are from")
    features: StudentFeatures
    imd_band: str | None = Field(None, examples=["10-20%"], description="Used by policy only")
    budget_left: float | None = Field(None, ge=0, description="Term budget left; default full")
    previous_contacts: int = Field(0, ge=0)


class Reason(BaseModel):
    feature: str
    contribution: float = Field(..., description="Added log-odds of failing or withdrawing")


class ScoreResponse(BaseModel):
    week: int
    risk: float = Field(..., ge=0, le=1, description="Calibrated P(fail or withdraw)")
    on_contact_list: bool = Field(..., description="Risk above the capacity threshold")
    risk_bucket: int
    top_reasons: list[Reason]
    recommended_action: Literal["none", "email", "call"]
    policy: str


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    path = artifacts_dir() / "risk_model.joblib"
    if not path.exists():
        raise RuntimeError(f"{path} not found. Run `fair-intervention train` first.")
    return ModelBundle.load(path)


@lru_cache(maxsize=1)
def get_policy() -> Policy:
    path = artifacts_dir() / "q_table.npy"
    return QLearningAgent.load(path) if path.exists() else ThresholdPolicy()


def policy_observation(
    risk: float, week: int, budget_left: float | None, imd_band: str | None, contacts: int
) -> np.ndarray:
    """Map an API request onto the simulator's observation.

    Simulator week 0 is the first early-warning week (course week 4).
    """
    cfg = EnvConfig()
    bucket = int(np.searchsorted(RISK_BUCKET_EDGES, risk, side="right"))
    env_week = min(week - HORIZONS[0], cfg.n_weeks - 1)
    budget = cfg.budget if budget_left is None else budget_left
    group = int(imd_band in LOW_IMD_BANDS)
    return np.array([bucket, env_week, budget_bin(budget, cfg), group, min(contacts, MAX_CONTACTS)])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/score", response_model=ScoreResponse)
def score(
    req: ScoreRequest,
    bundle: ModelBundle = Depends(get_bundle),
    policy: Policy = Depends(get_policy),
) -> ScoreResponse:
    model = bundle.models.get(req.week)
    if model is None:
        raise HTTPException(status_code=422, detail=f"no model for week {req.week}")
    row = pd.DataFrame([req.features.model_dump()]).astype(float)
    risk = float(model.predict_proba(row)[0])
    reasons = model.reasons(row)[0]
    obs = policy_observation(risk, req.week, req.budget_left, req.imd_band, req.previous_contacts)
    return ScoreResponse(
        week=req.week,
        risk=round(risk, 4),
        on_contact_list=risk >= bundle.thresholds[req.week],
        risk_bucket=int(obs[0]),
        top_reasons=[Reason(feature=f, contribution=round(c, 3)) for f, c in reasons],
        recommended_action=ACTION_NAMES[policy.act(obs)],
        policy=policy.name,
    )
