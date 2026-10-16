"""Temporal validation by presentation, grouped by student.

The model is trained on earlier presentations and tested on the latest one, because that is how
it would be used: fit on past cohorts, score the cohort that is running now. A random row split
would mix cohorts and hide drift between years.

OULAD students can retake a module, so the same ``id_student`` can appear in several
presentations. A retaker in the test cohort whose earlier attempt sits in training would let the
model recognise a person instead of a pattern, so those students are removed from the earlier
splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import HORIZONS, build_weekly_features


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    presentations: dict[str, list[str]]


def presentation_key(code: str) -> tuple[int, int]:
    """'2014B' -> (2014, 0); 'J' (October start) comes after 'B' (February start)."""
    return int(code[:4]), 0 if code[4] == "B" else 1


def temporal_split(frame: pd.DataFrame) -> Split:
    """Train on all but the last two presentations, validate on the second last, test on the last.

    Validation is used for probability calibration and for choosing decision thresholds, so the
    test cohort stays untouched until the final numbers.
    """
    ordered = sorted(frame["code_presentation"].unique(), key=presentation_key)
    if len(ordered) < 3:
        raise ValueError(f"need at least 3 presentations for a temporal split, got {ordered}")
    test_p, val_p, train_p = [ordered[-1]], [ordered[-2]], ordered[:-2]
    test = frame[frame["code_presentation"].isin(test_p)]
    val = frame[frame["code_presentation"].isin(val_p)]
    train = frame[frame["code_presentation"].isin(train_p)]
    val = val[~val["id_student"].isin(test["id_student"])]
    train = train[~train["id_student"].isin(pd.concat([val, test])["id_student"])]
    return Split(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        presentations={"train": list(train_p), "val": val_p, "test": test_p},
    )


def horizon_splits(
    tables: dict[str, pd.DataFrame], horizons: tuple[int, ...] = HORIZONS
) -> dict[int, Split]:
    """Features and temporal split for every early-warning week."""
    return {w: temporal_split(build_weekly_features(tables, w)) for w in horizons}
