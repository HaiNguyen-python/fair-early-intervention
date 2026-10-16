"""Synthetic data shape, the real-data loader, and the no-leakage guarantee of the features."""

import numpy as np
import pandas as pd
import pytest

from fair_intervention.data import oulad
from fair_intervention.data.synthetic import SyntheticConfig, generate
from fair_intervention.features import (
    FEATURES,
    KEY,
    PROTECTED,
    TARGET,
    build_weekly_features,
    imd_group,
    load_feature_mart,
)


def test_generator_is_seeded_and_follows_the_oulad_schema():
    cfg = SyntheticConfig(students_per_presentation=60, seed=11)
    a, b = generate(cfg), generate(cfg)
    for name, columns in oulad.REQUIRED_COLUMNS.items():
        assert set(columns) <= set(a[name].columns)
        pd.testing.assert_frame_equal(a[name], b[name])
    assert set(a["studentInfo"]["final_result"]) <= {"Pass", "Fail", "Withdrawn", "Distinction"}


def test_planted_effect_low_imd_students_fail_more_often(tables):
    info = tables["studentInfo"]
    at_risk = info["final_result"].isin(["Fail", "Withdrawn"])
    rates = at_risk.groupby(imd_group(info["imd_band"])).mean()
    assert rates["low"] > rates["high"] + 0.1


def _tamper_future(tables, cutoff):
    """Change everything that happens on or after the cut-off day."""
    t = {k: v.copy() for k, v in tables.items()}
    vle = t["studentVle"]
    vle.loc[vle["date"] >= cutoff, "sum_click"] *= 100
    sa = t["studentAssessment"]
    future = sa["date_submitted"] >= cutoff
    sa.loc[future, "score"] = 0
    sa.loc[future, "date_submitted"] = cutoff  # still in the future
    reg = t["studentRegistration"]
    reg.loc[reg["date_unregistration"] >= cutoff, "date_unregistration"] = cutoff + 50
    return t


@pytest.mark.parametrize("week", [4, 8])
def test_no_leakage_features_ignore_everything_after_the_cutoff(tables, week):
    before = build_weekly_features(tables, week)
    after = build_weekly_features(_tamper_future(tables, 7 * week), week)
    pd.testing.assert_frame_equal(before[KEY + FEATURES], after[KEY + FEATURES])


def test_features_drop_students_who_already_left_and_do_not_use_the_label(tables):
    week = 6
    frame = build_weekly_features(tables, week)
    reg = tables["studentRegistration"]
    gone = reg[reg["date_unregistration"] < 7 * week]
    merged = frame.merge(gone[KEY], on=KEY)
    assert merged.empty
    assert len(frame) == len(reg) - len(gone)
    assert TARGET not in FEATURES and not set(PROTECTED) & set(FEATURES)
    # Flipping every label changes the target column only.
    flipped = {k: v.copy() for k, v in tables.items()}
    flipped["studentInfo"]["final_result"] = "Pass"
    again = build_weekly_features(flipped, week)
    pd.testing.assert_frame_equal(frame[FEATURES], again[FEATURES])
    assert again[TARGET].sum() == 0


def test_missed_and_late_work_counts_on_a_tiny_example():
    key = {"code_module": "AAA", "code_presentation": "2014J"}
    tables = {
        "studentInfo": pd.DataFrame(
            [
                {
                    **key,
                    "id_student": 1,
                    "gender": "F",
                    "region": "Wales",
                    "highest_education": "A Level or Equivalent",
                    "imd_band": "10-20%",
                    "age_band": "0-35",
                    "num_of_prev_attempts": 0,
                    "studied_credits": 60,
                    "disability": "N",
                    "final_result": "Fail",
                },
            ]
        ),
        "studentRegistration": pd.DataFrame(
            [{**key, "id_student": 1, "date_registration": -5, "date_unregistration": np.nan}]
        ),
        "studentVle": pd.DataFrame(
            [
                {**key, "id_student": 1, "id_site": 9, "date": -2, "sum_click": 4},
                {**key, "id_student": 1, "id_site": 9, "date": 20, "sum_click": 6},
                {**key, "id_student": 1, "id_site": 8, "date": 27, "sum_click": 5},
                {**key, "id_student": 1, "id_site": 8, "date": 28, "sum_click": 99},
            ]
        ),
        "assessments": pd.DataFrame(
            [
                {**key, "id_assessment": 10, "assessment_type": "TMA", "date": 10, "weight": 10},
                {**key, "id_assessment": 11, "assessment_type": "CMA", "date": 20, "weight": 0},
                {**key, "id_assessment": 12, "assessment_type": "TMA", "date": 40, "weight": 20},
            ]
        ),
        "studentAssessment": pd.DataFrame(
            [
                {
                    "id_assessment": 10,
                    "id_student": 1,
                    "date_submitted": 12,
                    "is_banked": 0,
                    "score": 70.0,
                },
                # Submitted after the week-4 cut-off (day 28): unknown at week 4.
                {
                    "id_assessment": 11,
                    "id_student": 1,
                    "date_submitted": 30,
                    "is_banked": 0,
                    "score": 90.0,
                },
            ]
        ),
    }
    row = build_weekly_features(tables, 4).iloc[0]
    assert row["clicks_total"] == 15 and row["clicks_before_start"] == 4
    # Last week = days 21-27; day 20 belongs to the week before, day 28 is the future.
    assert row["clicks_last_week"] == 5 and row["days_since_last_click"] == 1
    assert (row["n_submitted"], row["n_late"], row["n_missed"]) == (1, 1, 1)
    assert row["mean_score"] == 70.0 and row["imd_group"] == "low"


def test_loader_round_trip_and_real_data_quirks(tmp_path):
    tables = generate(SyntheticConfig(students_per_presentation=30, seed=1))
    tables["studentInfo"].loc[0, "imd_band"] = "10-20"  # spelling used in the real files
    oulad.save(tables, tmp_path)
    loaded = oulad.load(tmp_path)
    assert loaded["studentInfo"].loc[0, "imd_band"] == "10-20%"
    assert len(loaded["studentVle"]) == len(tables["studentVle"])
    (tmp_path / "studentVle.csv").unlink()
    with pytest.raises(FileNotFoundError, match="generate"):
        oulad.load(tmp_path)


def test_validate_names_missing_columns():
    tables = generate(SyntheticConfig(students_per_presentation=20, seed=1))
    tables["studentVle"] = tables["studentVle"].drop(columns="sum_click")
    with pytest.raises(ValueError, match="sum_click"):
        oulad.validate(tables)


def test_feature_mart_contract(tmp_path, splits):
    path = tmp_path / "mart.csv"
    splits[4].test.to_csv(path, index=False)
    assert len(load_feature_mart(path)) == len(splits[4].test)
    splits[4].test.drop(columns="clicks_total").to_csv(path, index=False)
    with pytest.raises(ValueError, match="clicks_total"):
        load_feature_mart(path)
