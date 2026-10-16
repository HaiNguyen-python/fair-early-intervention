"""Group fairness metrics and a threshold-based mitigation.

Equal opportunity is the main criterion: among students who will actually fail or withdraw,
each group should have the same chance of being flagged (equal recall). For an intervention
programme this is the natural question: "do struggling students get help regardless of where
they live?" FPR and selection rate are reported next to it so that the cost is visible.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def group_metrics(y: np.ndarray, flagged: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Per-group size, base rate, selection rate, recall (TPR), FPR and precision."""
    df = pd.DataFrame(
        {"y": np.asarray(y).astype(bool), "f": np.asarray(flagged).astype(bool), "g": groups}
    )
    rows = []
    for g, d in df.groupby("g", sort=True):
        pos, neg = d["y"], ~d["y"]
        tp = int((d["f"] & pos).sum())
        fp = int((d["f"] & neg).sum())
        rows.append(
            {
                "group": g,
                "n": len(d),
                "positives": int(pos.sum()),
                "base_rate": pos.mean(),
                "selection_rate": d["f"].mean(),
                "recall": tp / pos.sum() if pos.sum() else np.nan,
                # Normal-approximation 95% half-width: small groups have wide recall error bars.
                "recall_ci": 1.96 * np.sqrt(tp * (pos.sum() - tp) / pos.sum() ** 3)
                if pos.sum()
                else np.nan,
                "fpr": fp / neg.sum() if neg.sum() else np.nan,
                "precision": tp / d["f"].sum() if d["f"].sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def equal_opportunity_gap(table: pd.DataFrame, min_positives: int = 50) -> float:
    """Largest minus smallest recall among groups with enough positives to measure it.

    Small groups are skipped: below about 50 positives the 95% interval on recall is wider than
    +/-14 points, so a "gap" there is mostly noise. Skipped groups still appear in the table.
    """
    measured = table[table["positives"] >= min_positives]["recall"]
    return float(measured.max() - measured.min()) if len(measured) > 1 else 0.0


def global_threshold(score: np.ndarray, capacity: float) -> float:
    """Score of the last student inside the top ``capacity`` share (flag ``score >= t``)."""
    k = math.ceil(capacity * len(score))
    return float(np.sort(np.asarray(score))[::-1][k - 1])


def _group_threshold(pos_scores: np.ndarray, target_recall: float) -> float:
    if len(pos_scores) == 0:
        return math.inf
    k = max(1, math.ceil(target_recall * len(pos_scores)))
    return float(np.sort(pos_scores)[::-1][k - 1])


def fit_group_thresholds(
    score: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    capacity: float,
    min_positives: int = 30,
    grid: int = 400,
) -> dict[str, float]:
    """Per-group thresholds that give every group the same recall within the same capacity.

    For a target recall r, each group's threshold is the score of its r-th best positive. The
    largest r whose combined list still fits ``capacity`` wins. Choose thresholds on validation
    data, then apply them unchanged to the test cohort.

    Groups with fewer than ``min_positives`` validation positives keep the global threshold:
    a threshold tuned on a handful of students would be noise. They are left out of the
    returned dict, so callers pass the global threshold as ``default`` when applying.

    This uses the group label at decision time. That is a policy choice, not only a technical
    one: it is the reason the mitigation works, and it must be agreed with the institution
    (see docs/model_card.md).
    """
    score, y, groups = np.asarray(score), np.asarray(y).astype(bool), np.asarray(groups)
    default = global_threshold(score, capacity)
    names = [g for g in sorted(set(groups.tolist())) if ((groups == g) & y).sum() >= min_positives]
    best = {g: default for g in names}
    for r in np.linspace(0, 1, grid + 1)[1:]:
        thresholds = {g: _group_threshold(score[(groups == g) & y], float(r)) for g in names}
        flagged = apply_group_thresholds(score, groups, thresholds, default)
        if flagged.mean() > capacity:
            break
        best = thresholds
    return best


def apply_group_thresholds(
    score: np.ndarray,
    groups: np.ndarray,
    thresholds: dict[str, float],
    default: float = math.inf,
) -> np.ndarray:
    """Flag ``score >= threshold[group]``; groups unseen in fitting use ``default``."""
    t = np.array([thresholds.get(g, default) for g in np.asarray(groups)])
    return np.asarray(score) >= t


def adjusted_scores(
    score: np.ndarray, groups: np.ndarray, thresholds: dict[str, float], default: float
) -> np.ndarray:
    """Shift each group's scores so that its own threshold lines up with the global one.

    Ranking the shifted scores and flagging the top K% applies the group thresholds while
    keeping the list size exactly at capacity, so "before" and "after" contact the same number
    of students and their precision can be compared fairly.
    """
    t = np.array([thresholds.get(g, default) for g in np.asarray(groups)])
    return np.asarray(score) - t + default
