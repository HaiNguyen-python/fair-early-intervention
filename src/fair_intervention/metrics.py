"""Metrics for an early-warning list that a school can only partly act on.

PR-AUC is used instead of ROC-AUC because staff care about the flagged list (precision) and the
students it finds (recall), not about true negatives. ``recall_at_capacity`` answers the
operational question: "if tutors can contact only the top K% of students, how many of the
students who will fail or withdraw do we reach?"
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score))


def flag_top_k(score: np.ndarray, capacity: float) -> np.ndarray:
    """Flag exactly ``ceil(capacity * n)`` students with the highest scores.

    A stable sort on the negated score breaks ties by row order, so the list size is exact and
    repeatable (a threshold on tied scores could overshoot the capacity).
    """
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    k = math.ceil(capacity * len(score))
    order = np.argsort(-np.asarray(score), kind="stable")
    flagged = np.zeros(len(score), dtype=bool)
    flagged[order[:k]] = True
    return flagged


def recall_at_capacity(y: np.ndarray, score: np.ndarray, capacity: float) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    flagged = flag_top_k(score, capacity)
    tp = int((flagged & y).sum())
    positives = int(y.sum())
    return {
        "recall": tp / positives if positives else 0.0,
        "precision": tp / int(flagged.sum()),
        "flagged": int(flagged.sum()),
        # Best possible recall with this list size: useful context when positives > capacity.
        "max_recall": min(1.0, int(flagged.sum()) / positives) if positives else 0.0,
    }


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean |predicted - observed| over equal-width probability bins."""
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            error += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(error)
