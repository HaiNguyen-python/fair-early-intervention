# Model card: early-warning risk model

Format follows Mitchell et al., "Model Cards for Model Reporting" (2019). Numbers are from the
synthetic run in [results.md](results.md) and must be replaced after a run on real data.

## Model details

- **What:** calibrated logistic regression (served) and HistGradientBoosting (comparison), one
  model per early-warning week (4, 6, 8).
- **Output:** probability that a student will fail or withdraw (`final_result` in Fail,
  Withdrawn), plus the three features that raise it most.
- **Calibration:** Platt scaling on the validation cohort.
- **Code:** `src/fair_intervention/risk.py`. Version 0.1.0. Author: Hai Nguyen.

## Intended use

- **For:** ranking enrolled students so tutors can offer **support** (an email, a call) to those
  most likely to struggle, within a fixed contact capacity.
- **Users:** student-support staff, with the reasons shown next to the score.
- **Not for:** grading, admissions, sanctions, or any decision that takes something away from a
  student. A false positive here costs one unneeded message; in those uses it would harm a
  person. The fairness criterion (equal recall) was chosen for the support use only.

## Inputs

Behaviour and study load known before day `7 * week`: VLE clicks and their trend, active days,
distinct resources, days since last click, clicks before the start, submitted / missed / late
coursework, mean score, previous attempts, credits. Gender, age band, IMD band and disability
are **not** inputs; they are used only to audit the model and, if the institution agrees, for
the group-threshold mitigation.

## Training and evaluation data

- **This repo:** synthetic data in the OULAD schema from `data/synthetic.py`, two modules, four
  presentations, 1,000 students each. Train: 2013B + 2013J; validation: 2014B; test: 2014J.
  Students who appear in a later split are removed from earlier ones.
- **Real data:** OULAD (Kuzilek et al., 2017), CC BY 4.0, loaded with `data/oulad.py`.
- **Planted effect:** low-IMD students in the synthetic data face outside shocks after the
  early-warning window. This is a test fixture for the fairness audit, not a claim about real
  students.

## Metrics (synthetic test cohort, week 6, capacity 20%)

| Metric | Logistic regression |
| --- | --- |
| PR-AUC | 0.702 |
| Recall@20% (max possible 50.9%) | 38.8% |
| Precision@20% | 76.3% |
| Expected calibration error | 0.025 |

Recall@capacity is the main metric because the decision is "who is on a list of fixed size".

## Fairness analysis

| Attribute | Equal-opportunity gap (recall, points) | Note |
| --- | --- | --- |
| IMD group | 15.3 | planted; low-IMD recall 30.3% vs 44.6-45.7% |
| Disability | 9.0 | not planted; 75 positives, ±10-point interval: mostly noise |
| Age band | 2.9 | not planted; the 55+ group is too small to measure |
| Gender | 2.3 | not planted |

**Mitigation:** group-specific thresholds for IMD groups, chosen on validation, applied at the
same 20% list size. Gap 15.3 → 3.3 points; low-IMD recall 30.3% → 37.7%; mid and high IMD
recall each fall about 5 points; overall precision unchanged (76.3%); low-IMD FPR 5.7% → 9.6%.

## Limitations and risks

- Synthetic results only. Real OULAD will give different numbers and possibly different gaps.
- One test cohort. Model metrics have sampling noise that is not shown as intervals.
- The model sees behaviour, and behaviour depends on circumstances. A quiet student may be
  working two jobs, not losing interest; reasons should be read as "what the data shows", not
  as a diagnosis.
- Group thresholds require the IMD band at decision time. Disability is special-category data
  under GDPR; using any group label in a decision needs a legal basis and the institution's
  agreement.
- Drift: a new course design or a new VLE changes what clicks mean. Re-check calibration and
  group recall every presentation.

## Ethical considerations

The model should widen access to help, not narrow it. Staff should be able to add students the
model missed, and students should know that support outreach is data-informed. The intervention
policy in `env.py` and `agents.py` is evaluated only in a simulator with assumed effect sizes and
must not be used to ration support for real students without off-policy evaluation on logged
data and a controlled pilot.
