"""Seeded synthetic data in the OULAD schema, with one fairness problem planted on purpose.

The real Open University Learning Analytics Dataset (OULAD) cannot be downloaded in CI, so this
module builds tables with the same names, columns and value formats. The generative story:

* Each student has a hidden engagement level and a hidden ability. Engagement drives VLE clicks
  and on-time submissions; ability drives scores. Both lower the chance of failing.
* **Planted fairness problem.** Students from the most deprived areas (``imd_band`` 0-30%) are
  far more likely to hit an outside shock (a job, care duties) *after* the early-warning window.
  The shock causes withdrawal or failure, but it is invisible in the first weeks of click data.
  A model that only sees behaviour therefore ranks these students too low: it has lower recall
  for the low-IMD group. This mirrors a real concern in learning analytics: the same behaviour
  can mean different things for students with different circumstances.
* Gender, age band and disability have no planted effect. They act as a control: their gaps
  should be small and noisy.

All effect sizes are invented. They make the pipeline's behaviour visible; they say nothing about
real students.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MODULES = ("AAA", "BBB")
# OULAD presentation codes: year + B (February start) or J (October start).
PRESENTATIONS = ("2013B", "2013J", "2014B", "2014J")
TERM_DAYS = 210
IMD_BANDS = (
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
)
LOW_IMD_BANDS = IMD_BANDS[:3]
REGIONS = (
    "East Anglian Region",
    "Scotland",
    "North Western Region",
    "South East Region",
    "West Midlands Region",
    "Wales",
    "North Region",
    "South Region",
    "Ireland",
    "South West Region",
    "East Midlands Region",
    "Yorkshire Region",
    "London Region",
)
EDUCATION = (
    "No Formal quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
)
EDUCATION_P = (0.02, 0.38, 0.42, 0.15, 0.03)
AGE_BANDS = ("0-35", "35-55", "55<=")
AGE_P = (0.70, 0.28, 0.02)
CREDITS = (30, 60, 60, 60, 90, 120)
# Coursework schedule per presentation: (type, due day, weight).
SCHEDULE = (
    ("CMA", 12, 0.0),
    ("TMA", 19, 10.0),
    ("CMA", 33, 0.0),
    ("TMA", 47, 15.0),
    ("TMA", 82, 20.0),
    ("TMA", 117, 25.0),
    ("TMA", 152, 30.0),
    ("Exam", 205, 100.0),
)
SITES_PER_MODULE = 40


@dataclass(frozen=True)
class SyntheticConfig:
    students_per_presentation: int = 1000
    seed: int = 42
    shock_rate_low_imd: float = 0.45
    shock_rate_other: float = 0.06
    retake_share: float = 0.06
    missing_imd_share: float = 0.04


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _students(
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    module: str,
    presentation: str,
    next_id: int,
    retake_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Demographics and hidden traits for one presentation. Retakers keep their id and profile."""
    n = cfg.students_per_presentation
    n_retake = min(int(round(n * cfg.retake_share)), len(retake_pool))
    retakers = retake_pool.sample(n=n_retake, random_state=int(rng.integers(1 << 31)))
    n_new = n - n_retake
    imd = rng.choice(np.array(IMD_BANDS, dtype=object), size=n_new)
    imd[rng.random(n_new) < cfg.missing_imd_share] = np.nan
    new = pd.DataFrame(
        {
            "id_student": next_id + np.arange(n_new) * 3 + rng.integers(0, 3, n_new),
            "gender": rng.choice(["M", "F"], size=n_new, p=[0.55, 0.45]),
            "region": rng.choice(REGIONS, size=n_new),
            "highest_education": rng.choice(EDUCATION, size=n_new, p=EDUCATION_P),
            "imd_band": imd,
            "age_band": rng.choice(AGE_BANDS, size=n_new, p=AGE_P),
            "num_of_prev_attempts": 0,
            "disability": rng.choice(["N", "Y"], size=n_new, p=[0.9, 0.1]),
        }
    )
    parts = [new]
    if n_retake:
        old = retakers[new.columns].assign(
            num_of_prev_attempts=lambda d: d["num_of_prev_attempts"] + 1
        )
        parts.append(old)
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "code_presentation", presentation)
    df.insert(0, "code_module", module)
    df["studied_credits"] = rng.choice(CREDITS, size=n)
    return df, int(next_id + n_new * 3 + 3)


def _outcomes(rng: np.random.Generator, cfg: SyntheticConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Hidden traits, the planted shock, final_result and the withdrawal day."""
    n = len(df)
    engagement = rng.normal(size=n)
    ability = 0.4 * engagement + np.sqrt(1 - 0.16) * rng.normal(size=n)
    low_imd = df["imd_band"].isin(LOW_IMD_BANDS).to_numpy()
    shock_rate = np.where(low_imd, cfg.shock_rate_low_imd, cfg.shock_rate_other)
    shock = rng.random(n) < shock_rate
    # The shock lands after week 6, beyond what an early-warning model can observe.
    shock_day = np.where(shock, rng.integers(45, 140, n), TERM_DAYS + 1)
    logit = (
        -1.1
        - 1.3 * engagement
        - 0.8 * ability
        + 0.4 * df["num_of_prev_attempts"].to_numpy()
        + 0.006 * (df["studied_credits"].to_numpy() - 60)
        + 3.0 * shock
    )
    at_risk = rng.random(n) < _sigmoid(logit)
    p_withdraw = np.where(shock, 0.7, _sigmoid(-0.2 - 0.8 * engagement))
    withdrawn = at_risk & (rng.random(n) < p_withdraw)
    distinction = ~at_risk & (ability > 1.0) & (rng.random(n) < 0.6)
    result = np.select(
        [withdrawn, at_risk, distinction], ["Withdrawn", "Fail", "Distinction"], "Pass"
    )
    withdraw_day = np.where(
        shock, shock_day + rng.integers(0, 21, n), rng.integers(10, 170, n)
    ).astype(float)
    withdraw_day[~withdrawn] = np.nan
    return df.assign(
        final_result=result,
        _engagement=engagement,
        _ability=ability,
        _shock_day=shock_day,
        _withdraw_day=withdraw_day,
        _at_risk=at_risk,
    )


def _vle(rng: np.random.Generator, df: pd.DataFrame, site_ids: np.ndarray) -> pd.DataFrame:
    """Daily click rows. Engagement sets the level; failing students fade; shocks cut clicks."""
    days = np.arange(-10, TERM_DAYS)
    e = df["_engagement"].to_numpy()[:, None]
    fade = np.where(df["_at_risk"].to_numpy()[:, None], np.exp(-0.004 * np.maximum(days, 0)), 1.0)
    shocked = days[None, :] >= df["_shock_day"].to_numpy()[:, None]
    p_active = _sigmoid(-0.3 + 0.9 * e) * fade * np.where(shocked, 0.3, 1.0)
    p_active = np.where(days[None, :] < 0, p_active * 0.3, p_active)
    withdraw = np.nan_to_num(df["_withdraw_day"].to_numpy(), nan=np.inf)[:, None]
    active = (rng.random(p_active.shape) < p_active) & (days[None, :] < withdraw)
    rows, cols = np.nonzero(active)
    mu = 4.0 * np.exp(0.5 * df["_engagement"].to_numpy()[rows])
    return pd.DataFrame(
        {
            "code_module": df["code_module"].to_numpy()[rows],
            "code_presentation": df["code_presentation"].to_numpy()[rows],
            "id_student": df["id_student"].to_numpy()[rows],
            "id_site": rng.choice(site_ids, size=len(rows)),
            "date": days[cols],
            "sum_click": rng.poisson(mu) + 1,
        }
    )


def _submissions(
    rng: np.random.Generator, df: pd.DataFrame, assessments: pd.DataFrame
) -> pd.DataFrame:
    """One row per submitted assessment. Low engagement means missed or late work."""
    parts = []
    e = df["_engagement"].to_numpy()
    a = df["_ability"].to_numpy()
    withdraw = np.nan_to_num(df["_withdraw_day"].to_numpy(), nan=np.inf)
    for row in assessments.itertuples():
        n = len(df)
        if row.assessment_type == "Exam":
            submit = ~df["_at_risk"].to_numpy() | (rng.random(n) < 0.5)
            lateness = np.zeros(n, dtype=int)
        else:
            shocked = row.date >= df["_shock_day"].to_numpy()
            submit = rng.random(n) < _sigmoid(1.6 + 1.2 * e - 2.5 * shocked)
            lateness = np.round(rng.normal(-1.0 - 1.5 * e, 3.0)).astype(int)
        submitted_on = row.date + lateness
        submit &= submitted_on < withdraw
        score = np.clip(np.round(62 + 12 * a + 5 * e + rng.normal(0, 10, n)), 0, 100)
        parts.append(
            pd.DataFrame(
                {
                    "id_assessment": row.id_assessment,
                    "id_student": df["id_student"].to_numpy()[submit],
                    "date_submitted": submitted_on[submit],
                    "is_banked": 0,
                    "score": score[submit],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def generate(cfg: SyntheticConfig | None = None) -> dict[str, pd.DataFrame]:
    """Return the OULAD tables keyed by their CSV file stem (``studentInfo``, ``studentVle``, ...).

    Presentations are generated in time order so that retakers in a later presentation are
    students who failed or withdrew earlier, as in the real data.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    info_parts, vle_parts, sa_parts, assess_parts = [], [], [], []
    next_id, next_assessment = 10_000, 1_000
    for m_idx, module in enumerate(MODULES):
        site_ids = 500_000 + m_idx * 1_000 + np.arange(SITES_PER_MODULE)
        pool = pd.DataFrame()
        for presentation in PRESENTATIONS:
            students, next_id = _students(rng, cfg, module, presentation, next_id, pool)
            students = _outcomes(rng, cfg, students)
            assessments = pd.DataFrame(
                {
                    "code_module": module,
                    "code_presentation": presentation,
                    "id_assessment": next_assessment + np.arange(len(SCHEDULE)),
                    "assessment_type": [s[0] for s in SCHEDULE],
                    "date": [float(s[1]) for s in SCHEDULE],
                    "weight": [s[2] for s in SCHEDULE],
                }
            )
            next_assessment += len(SCHEDULE)
            info_parts.append(students)
            assess_parts.append(assessments)
            vle_parts.append(_vle(rng, students, site_ids))
            sa_parts.append(_submissions(rng, students, assessments))
            # Students who failed or withdrew may come back; anyone who passed leaves the pool.
            is_fail = students["final_result"].isin(["Fail", "Withdrawn"])
            pool = pd.concat([pool, students[is_fail]], ignore_index=True)
            pool = pool.drop_duplicates("id_student", keep="last")
            pool = pool[~pool["id_student"].isin(students.loc[~is_fail, "id_student"])]
    info = pd.concat(info_parts, ignore_index=True)
    registration = info[["code_module", "code_presentation", "id_student"]].assign(
        date_registration=rng.integers(-60, 0, len(info)).astype(float),
        date_unregistration=info["_withdraw_day"],
    )
    student_info = info[[c for c in info.columns if not c.startswith("_")]]
    return {
        "studentInfo": student_info.reset_index(drop=True),
        "studentRegistration": registration.reset_index(drop=True),
        "studentVle": pd.concat(vle_parts, ignore_index=True),
        "studentAssessment": pd.concat(sa_parts, ignore_index=True),
        "assessments": pd.concat(assess_parts, ignore_index=True),
    }
