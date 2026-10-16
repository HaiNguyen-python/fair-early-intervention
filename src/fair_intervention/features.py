"""Weekly student features "as of week N", built only from what was known at that moment.

Leakage rule: at week N the cut-off is day ``7 * N`` (OULAD days count from the module start).
Only VLE rows with ``date < cutoff``, submissions with ``date_submitted < cutoff`` and
assessments due before the cut-off are used. Students who already unregistered before the
cut-off are dropped: they are gone, so there is nobody left to contact.
``tests/test_features.py`` changes every future row and checks that no feature moves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data.synthetic import LOW_IMD_BANDS

KEY = ["code_module", "code_presentation", "id_student"]
HORIZONS = (4, 6, 8)
FEATURES = [
    "clicks_total",
    "clicks_last_week",
    "clicks_trend",
    "active_days",
    "distinct_sites",
    "days_since_last_click",
    "clicks_before_start",
    "n_submitted",
    "n_missed",
    "n_late",
    "mean_score",
    "num_of_prev_attempts",
    "studied_credits",
]
# Kept for fairness analysis only; the risk model never sees them ("fairness through
# unawareness", which the fairness report shows is not enough on its own).
PROTECTED = ["gender", "age_band", "imd_group", "disability"]
TARGET = "at_risk"


def imd_group(band: pd.Series) -> pd.Series:
    """Collapse ten IMD deciles into three groups big enough to measure recall on."""
    low = band.isin(LOW_IMD_BANDS)
    high = band.isin(["70-80%", "80-90%", "90-100%"])
    out = np.select([band.isna(), low, high], ["unknown", "low", "high"], "mid")
    return pd.Series(out, index=band.index)


def _vle_features(vle: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    known = vle[vle["date"] < cutoff]
    g = known.groupby(KEY)
    last_week = known[known["date"] >= cutoff - 7].groupby(KEY)["sum_click"].sum()
    prev_week = known[(known["date"] >= cutoff - 14) & (known["date"] < cutoff - 7)]
    prev = prev_week.groupby(KEY)["sum_click"].sum()
    out = pd.DataFrame(
        {
            "clicks_total": g["sum_click"].sum(),
            "active_days": g["date"].nunique(),
            "distinct_sites": g["id_site"].nunique(),
            "last_click_day": g["date"].max(),
            "clicks_before_start": known[known["date"] < 0].groupby(KEY)["sum_click"].sum(),
            "clicks_last_week": last_week,
            "clicks_prev_week": prev,
        }
    )
    return out.reset_index()


def _assessment_features(
    sa: pd.DataFrame, assessments: pd.DataFrame, students: pd.DataFrame, cutoff: int
) -> pd.DataFrame:
    # The exam is at the end of term; it can never be known in an early-warning window.
    due = assessments[(assessments["date"] < cutoff) & (assessments["assessment_type"] != "Exam")]
    n_due = due.groupby(["code_module", "code_presentation"]).size().rename("n_due").reset_index()
    subs = sa[sa["date_submitted"] < cutoff].merge(
        due[["id_assessment", "code_module", "code_presentation", "date"]], on="id_assessment"
    )
    g = subs.assign(late=subs["date_submitted"] > subs["date"]).groupby(KEY)
    per_student = pd.DataFrame(
        {"n_submitted": g.size(), "n_late": g["late"].sum(), "mean_score": g["score"].mean()}
    ).reset_index()
    out = students[KEY].merge(n_due, on=["code_module", "code_presentation"], how="left")
    out = out.merge(per_student, on=KEY, how="left")
    out[["n_due", "n_submitted", "n_late"]] = out[["n_due", "n_submitted", "n_late"]].fillna(0)
    out["n_missed"] = out["n_due"] - out["n_submitted"]
    return out.drop(columns="n_due")


def build_weekly_features(tables: dict[str, pd.DataFrame], week: int) -> pd.DataFrame:
    """One row per student still registered at the start of ``week``, with features and label.

    Returns ``KEY + FEATURES + PROTECTED + [TARGET, "week"]``. ``mean_score`` is NaN for students
    with no submitted work yet; models must handle that (both pipelines do).
    """
    cutoff = 7 * week
    info = tables["studentInfo"]
    reg = tables["studentRegistration"]
    gone = reg.loc[reg["date_unregistration"] < cutoff, KEY]
    students = info.merge(gone, on=KEY, how="left", indicator=True)
    students = students[students["_merge"] == "left_only"].drop(columns="_merge")

    vle = _vle_features(tables["studentVle"], cutoff)
    sa = _assessment_features(tables["studentAssessment"], tables["assessments"], students, cutoff)
    df = students.merge(vle, on=KEY, how="left").merge(sa, on=KEY, how="left")
    counts = ["clicks_total", "active_days", "distinct_sites", "clicks_before_start"]
    counts += ["clicks_last_week", "clicks_prev_week"]
    df[counts] = df[counts].fillna(0)
    # Never clicked: treat as "silent since before the course started".
    df["days_since_last_click"] = cutoff - df["last_click_day"].fillna(-1)
    df["clicks_trend"] = np.log1p(df["clicks_last_week"]) - np.log1p(df["clicks_prev_week"])
    df["imd_group"] = imd_group(df["imd_band"])
    df[TARGET] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    df["week"] = week
    cols = KEY + FEATURES + PROTECTED + [TARGET, "week"]
    return df[cols].sort_values(KEY).reset_index(drop=True)


def load_feature_mart(path: Path) -> pd.DataFrame:
    """Read a weekly feature mart exported by the companion ``oulad-lakehouse`` repo (CSV).

    The mart must follow this module's column contract; checking it here turns a silent
    schema drift between the two repos into a clear error.
    """
    df = pd.read_csv(path)
    missing = set(KEY + FEATURES + PROTECTED + [TARGET, "week"]) - set(df.columns)
    if missing:
        raise ValueError(f"feature mart {path} is missing columns: {sorted(missing)}")
    return df
