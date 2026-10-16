"""Risk models, the fairness maths, and the mitigation on the planted problem."""

import numpy as np
import pytest

from fair_intervention.data.synthetic import SyntheticConfig, generate
from fair_intervention.fairness import (
    adjusted_scores,
    apply_group_thresholds,
    equal_opportunity_gap,
    fit_group_thresholds,
    global_threshold,
    group_metrics,
)
from fair_intervention.features import TARGET, build_weekly_features
from fair_intervention.metrics import flag_top_k, pr_auc
from fair_intervention.risk import ModelBundle, fit_bundle, fit_risk_model
from fair_intervention.split import temporal_split


def test_group_metrics_hand_computed():
    #          group a: 3 positives, 2 negatives | group b: 2 positives, 3 negatives
    y = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    flagged = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0]).astype(bool)
    groups = np.array(["a"] * 5 + ["b"] * 5)
    t = group_metrics(y, flagged, groups).set_index("group")
    assert t.loc["a", "recall"] == pytest.approx(2 / 3)
    assert t.loc["a", "fpr"] == pytest.approx(1 / 2)
    assert t.loc["a", "selection_rate"] == pytest.approx(3 / 5)
    assert t.loc["a", "precision"] == pytest.approx(2 / 3)
    assert t.loc["b", "recall"] == pytest.approx(1 / 2)
    assert t.loc["b", "fpr"] == 0.0
    assert t.loc["b", "selection_rate"] == pytest.approx(1 / 5)
    assert equal_opportunity_gap(t.reset_index(), min_positives=1) == pytest.approx(1 / 6)
    # Groups below the size floor are not counted: no gap to report.
    assert equal_opportunity_gap(t.reset_index(), min_positives=3) == 0.0


def test_group_thresholds_equalise_recall_within_capacity():
    rng = np.random.default_rng(0)
    groups = np.array(["a"] * 500 + ["b"] * 500)
    y = rng.random(1000) < 0.4
    # Group b's positives score lower: a model that under-ranks one group.
    score = rng.normal(size=1000) + 1.5 * y - 0.8 * (y & (groups == "b"))
    th = fit_group_thresholds(score, y, groups, capacity=0.2)
    flagged = apply_group_thresholds(score, groups, th)
    assert flagged.mean() <= 0.2
    after = group_metrics(y, flagged, groups)
    before = group_metrics(y, score >= global_threshold(score, 0.2), groups)
    assert equal_opportunity_gap(after) < 0.03 < equal_opportunity_gap(before)
    assert th["b"] < th["a"]


def test_small_groups_keep_the_global_threshold_and_unseen_groups_use_default():
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    y = np.array([1, 0, 1, 0, 1, 0]).astype(bool)
    groups = np.array(["a", "a", "a", "a", "a", "tiny"])
    th = fit_group_thresholds(score, y, groups, capacity=0.5, min_positives=2)
    assert "tiny" not in th
    shifted = adjusted_scores(score, groups, th, default=0.7)
    assert shifted[-1] == score[-1]
    assert not apply_group_thresholds(np.array([0.99]), np.array(["new"]), th)[0]
    assert apply_group_thresholds(np.array([0.99]), np.array(["new"]), th, 0.5)[0]


def test_models_beat_the_base_rate_and_calibration_keeps_the_ranking(splits):
    s = splits[6]
    for kind in ("logistic", "hist_gb"):
        model = fit_risk_model(s.train, s.val, kind, 6)
        raw, cal = model.raw_proba(s.test), model.predict_proba(s.test)
        assert np.array_equal(np.argsort(raw, kind="stable"), np.argsort(cal, kind="stable"))
        assert pr_auc(s.test[TARGET].to_numpy(), cal) > s.test[TARGET].mean() + 0.15
    with pytest.raises(ValueError):
        fit_risk_model(s.train, s.val, "svm", 6)  # type: ignore[arg-type]


def test_threshold_mitigation_reduces_the_imd_gap_on_synthetic_data():
    # The full default dataset: group recall needs a few hundred positives per group.
    s = temporal_split(build_weekly_features(generate(SyntheticConfig()), 6))
    model = fit_risk_model(s.train, s.val, "logistic", 6)
    p_val, p_test = model.predict_proba(s.val), model.predict_proba(s.test)
    y = s.test[TARGET].to_numpy()
    g_val, g_test = s.val["imd_group"].to_numpy(), s.test["imd_group"].to_numpy()
    before = group_metrics(y, flag_top_k(p_test, 0.2), g_test)
    th = fit_group_thresholds(p_val, s.val[TARGET].to_numpy(), g_val, 0.2)
    shifted = adjusted_scores(p_test, g_test, th, global_threshold(p_val, 0.2))
    after = group_metrics(y, flag_top_k(shifted, 0.2), g_test)
    low = lambda t: t.set_index("group").loc["low", "recall"]  # noqa: E731
    # The planted problem is visible, and the mitigation shrinks it.
    assert low(before) < before.set_index("group").loc["high", "recall"]
    assert equal_opportunity_gap(after) < equal_opportunity_gap(before)
    assert low(after) > low(before)


def test_reasons_explain_the_logistic_log_odds(splits, tmp_path):
    bundle = fit_bundle({6: splits[6]}, 0.2, "test")
    path = tmp_path / "m.joblib"
    bundle.save(path)
    model = ModelBundle.load(path).models[6]
    reasons = model.reasons(splits[6].test.head(5))
    assert all(len(r) <= 3 and all(c > 0 for _, c in r) for r in reasons)
    hgb = fit_risk_model(splits[6].train, splits[6].val, "hist_gb", 6)
    with pytest.raises(ValueError):
        hgb.reasons(splits[6].test.head(1))
