"""Risk models: logistic regression baseline and histogram gradient boosting, both calibrated.

Both models see only behaviour and study-load features (``features.FEATURES``), never the
protected attributes. Probabilities are calibrated on the validation cohort with Platt scaling,
so that "0.6" means roughly "6 in 10 students like this fail or withdraw". Platt scaling is
monotone, so it changes probabilities but not the ranking that the capacity metric uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .fairness import global_threshold
from .features import FEATURES, TARGET
from .metrics import expected_calibration_error, pr_auc, recall_at_capacity
from .split import Split

ModelKind = Literal["logistic", "hist_gb"]
EPS = 1e-6


def make_estimator(kind: ModelKind, seed: int = 0) -> Pipeline | HistGradientBoostingClassifier:
    if kind == "logistic":
        return Pipeline(
            [
                # add_indicator keeps "no work submitted yet" as its own signal.
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=1000)),
            ]
        )
    if kind == "hist_gb":
        # Small, regularised trees: a few thousand rows per horizon do not support deep ones.
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=seed
        )
    raise ValueError(f"unknown model kind {kind!r}")


@dataclass
class PlattCalibrator:
    """Logistic regression on the logit of the raw score."""

    model: LogisticRegression = field(default_factory=LogisticRegression)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self.model.fit(self._logit(raw), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(raw))[:, 1]


@dataclass
class RiskModel:
    kind: ModelKind
    week: int
    estimator: Pipeline | HistGradientBoostingClassifier
    calibrator: PlattCalibrator
    features: list[str]

    def raw_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(frame[self.features])[:, 1]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.raw_proba(frame))

    def reasons(self, frame: pd.DataFrame, top: int = 3) -> list[list[tuple[str, float]]]:
        """Top features pushing each student's risk up: coefficient x standardised value.

        Exact for the logistic model (its log-odds is the sum of these terms), which is why the
        API serves that model. Tree models have no such decomposition without extra tooling.
        """
        if self.kind != "logistic":
            raise ValueError("per-student reasons are only defined for the logistic model")
        assert isinstance(self.estimator, Pipeline)
        prep = self.estimator[:-1]
        coef = self.estimator[-1].coef_[0]
        names = [str(n).replace("missingindicator_", "no_") for n in prep.get_feature_names_out()]
        contributions = prep.transform(frame[self.features]) * coef
        out = []
        for row in contributions:
            order = np.argsort(-row)[:top]
            out.append([(names[i], float(row[i])) for i in order if row[i] > 0])
        return out


def fit_risk_model(
    train: pd.DataFrame, val: pd.DataFrame, kind: ModelKind, week: int, seed: int = 0
) -> RiskModel:
    """Fit on the training cohorts, calibrate on the validation cohort."""
    estimator = make_estimator(kind, seed)
    estimator.fit(train[FEATURES], train[TARGET])
    raw_val = estimator.predict_proba(val[FEATURES])[:, 1]
    calibrator = PlattCalibrator().fit(raw_val, val[TARGET].to_numpy())
    return RiskModel(kind, week, estimator, calibrator, list(FEATURES))


def evaluate(model: RiskModel, frame: pd.DataFrame, capacity: float) -> dict[str, float]:
    y = frame[TARGET].to_numpy()
    raw = model.raw_proba(frame)
    prob = model.calibrator.transform(raw)
    at_cap = recall_at_capacity(y, prob, capacity)
    return {
        "n": float(len(frame)),
        "base_rate": float(y.mean()),
        "pr_auc": pr_auc(y, prob),
        "recall_at_capacity": at_cap["recall"],
        "precision_at_capacity": at_cap["precision"],
        "max_recall": at_cap["max_recall"],
        "ece_raw": expected_calibration_error(y, raw),
        "ece_calibrated": expected_calibration_error(y, prob),
    }


@dataclass
class ModelBundle:
    """What the API needs: one calibrated model per horizon and its capacity threshold.

    The threshold is the calibrated risk of the last student inside the top ``capacity``
    share of the validation cohort, so ``risk >= threshold`` approximates "would be on the
    contact list".
    """

    models: dict[int, RiskModel]
    thresholds: dict[int, float]
    capacity: float
    data_label: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> ModelBundle:
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle


def fit_bundle(splits: dict[int, Split], capacity: float, data_label: str) -> ModelBundle:
    models, thresholds = {}, {}
    for week, split in splits.items():
        model = fit_risk_model(split.train, split.val, "logistic", week)
        models[week] = model
        thresholds[week] = global_threshold(model.predict_proba(split.val), capacity)
    return ModelBundle(models, thresholds, capacity, data_label)
