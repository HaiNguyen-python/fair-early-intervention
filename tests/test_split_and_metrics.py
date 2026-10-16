"""Temporal split correctness and the metric maths on hand-computed examples."""

import numpy as np
import pandas as pd
import pytest

from fair_intervention.metrics import (
    expected_calibration_error,
    flag_top_k,
    recall_at_capacity,
)
from fair_intervention.split import presentation_key, temporal_split


def test_presentations_sort_by_year_then_february_before_october():
    codes = ["2014J", "2013J", "2014B", "2013B"]
    assert sorted(codes, key=presentation_key) == ["2013B", "2013J", "2014B", "2014J"]


def test_temporal_split_trains_on_the_past_and_never_shares_students(splits):
    s = splits[6]
    assert s.presentations == {"train": ["2013B", "2013J"], "val": ["2014B"], "test": ["2014J"]}
    assert set(s.test["code_presentation"]) == {"2014J"}
    ids = [set(s.train["id_student"]), set(s.val["id_student"]), set(s.test["id_student"])]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]


def test_temporal_split_removes_a_retaker_from_training():
    frame = pd.DataFrame(
        {
            "code_presentation": ["2013B", "2013B", "2013J", "2014B", "2014J"],
            "id_student": [1, 2, 3, 4, 1],  # student 1 retakes in the test presentation
        }
    )
    s = temporal_split(frame)
    assert sorted(s.train["id_student"]) == [2, 3]
    assert list(s.test["id_student"]) == [1]


def test_temporal_split_needs_three_presentations():
    with pytest.raises(ValueError, match="3 presentations"):
        temporal_split(pd.DataFrame({"code_presentation": ["2013B"], "id_student": [1]}))


def test_recall_at_capacity_hand_computed():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    # 30% of 10 = 3 students: rows 0, 1, 2 -> 2 of the 3 positives.
    r = recall_at_capacity(y, score, 0.3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["flagged"] == 3 and r["max_recall"] == 1.0
    # 25% of 10 rounds up to 3, never down: the list size is a promise to tutors.
    assert flag_top_k(score, 0.25).sum() == 3


def test_flag_top_k_is_exact_with_ties():
    flagged = flag_top_k(np.array([0.5, 0.5, 0.5, 0.5]), 0.5)
    assert flagged.tolist() == [True, True, False, False]
    with pytest.raises(ValueError):
        flag_top_k(np.array([0.1]), 0)


def test_expected_calibration_error_hand_computed():
    # Bin 0.2: predicted 0.2, observed 0.5 -> 0.3; bin 0.8: predicted 0.8, observed 1.0 -> 0.2.
    prob = np.array([0.2, 0.2, 0.8, 0.8])
    y = np.array([0, 1, 1, 1])
    assert expected_calibration_error(y, prob) == pytest.approx(0.5 * 0.3 + 0.5 * 0.2)
    assert expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
