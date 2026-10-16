# Results

Data: **Synthetic OULAD-shaped data**. Every number below is produced by `fair-intervention report`; nothing is copied from elsewhere.

Temporal split: train on 2013B, 2013J, calibrate and choose thresholds on 2014B, test on 2014J. Students who appear in a later split are removed from the earlier ones.

## 1. Risk model (test cohort)

Contact capacity: tutors can reach the top 20% of enrolled students. `max recall` is the best any model could do with a list that size.

| Week | Model | Students | At-risk rate | PR-AUC | Recall@20% | Max recall | Precision@20% | ECE raw → calibrated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | Logistic regression | 1958 | 40.0% | 0.690 | 38.0% | 50.0% | 76.0% | 0.038 → 0.036 |
| 4 | Hist. gradient boosting | 1958 | 40.0% | 0.678 | 37.8% | 50.0% | 75.5% | 0.045 → 0.039 |
| 6 | Logistic regression | 1937 | 39.4% | 0.702 | 38.8% | 50.9% | 76.3% | 0.025 → 0.025 |
| 6 | Hist. gradient boosting | 1937 | 39.4% | 0.693 | 37.4% | 50.9% | 73.5% | 0.031 → 0.022 |
| 8 | Logistic regression | 1906 | 38.4% | 0.730 | 41.1% | 52.2% | 78.8% | 0.026 → 0.024 |
| 8 | Hist. gradient boosting | 1906 | 38.4% | 0.728 | 40.8% | 52.2% | 78.3% | 0.027 → 0.026 |

## 2. Fairness at week 6 (logistic model, top 20% list)

Recall = share of students who later fail or withdraw that the list reaches. ± is a 95% interval. The equal-opportunity gap only counts groups with at least 50 positives.

**gender** (equal-opportunity gap: 2.3 points)

| Group | Students | Positives | Base rate | Selected | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| F | 848 | 319 | 37.6% | 19.8% | 40.1% ± 5.4% | 7.6% |
| M | 1089 | 444 | 40.8% | 20.2% | 37.8% ± 4.5% | 8.1% |

**age_band** (equal-opportunity gap: 2.9 points)

| Group | Students | Positives | Base rate | Selected | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| 0-35 | 1360 | 525 | 38.6% | 20.0% | 37.9% ± 4.2% | 8.7% |
| 35-55 | 544 | 228 | 41.9% | 20.6% | 40.8% ± 6.4% | 6.0% |
| 55<= | 33 | 10 | 30.3% | 12.1% | 40.0% ± 30.4% | 0.0% |

**imd_group** (equal-opportunity gap: 15.3 points)

| Group | Students | Positives | Base rate | Selected | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| high | 551 | 177 | 32.1% | 20.7% | 44.6% ± 7.3% | 9.4% |
| low | 581 | 300 | 51.6% | 18.4% | 30.3% ± 5.2% | 5.7% |
| mid | 721 | 254 | 35.2% | 21.2% | 45.7% ± 6.1% | 7.9% |
| unknown | 84 | 32 | 38.1% | 16.7% | 31.2% ± 16.1% | 7.7% |

**disability** (equal-opportunity gap: 9.0 points)

| Group | Students | Positives | Base rate | Selected | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| N | 1739 | 688 | 39.6% | 20.1% | 39.7% ± 3.7% | 7.2% |
| Y | 198 | 75 | 37.9% | 19.7% | 30.7% ± 10.4% | 13.0% |

## 3. Mitigation: group thresholds for IMD groups

Thresholds chosen on the validation cohort, applied to the test cohort. Both lists contact exactly 20% of students, so precision is comparable.

| Metric | One global threshold | Group thresholds |
| --- | --- | --- |
| Overall precision | 76.3% | 76.3% |
| Overall recall | 38.8% | 38.8% |
| Equal-opportunity gap (IMD), points | 15.3 | 3.3 |
| Recall, High IMD (70-100%) | 44.6% | 39.0% |
| FPR, High IMD (70-100%) | 9.4% | 7.8% |
| Recall, Low IMD (0-30%) | 30.3% | 37.7% |
| FPR, Low IMD (0-30%) | 5.7% | 9.6% |
| Recall, Mid IMD (30-70%) | 45.7% | 40.9% |
| FPR, Mid IMD (30-70%) | 7.9% | 6.9% |
| Recall, IMD unknown | 31.2% | 31.2% |
| FPR, IMD unknown | 7.7% | 7.7% |

## 4. Intervention policies (simulator)

300 evaluation terms of 40 students, 8 weeks, budget 40 (email 1, call 5). Mean ± 95% interval. Objective = retained − 0.02 × spend (λ = 0). The simulator's effect sizes are assumptions, see docs/decisions.md.

| Policy | Retained | Spent | Coverage gap (points) | Objective |
| --- | --- | --- | --- | --- |
| Never intervene | 26.46 ± 0.34 | 0.0 ± 0.0 | 0.0 ± 0.0 | 26.46 ± 0.34 |
| Random | 27.96 ± 0.32 | 40.0 ± 0.0 | 17.1 ± 1.5 | 27.16 ± 0.32 |
| Threshold rule | 28.41 ± 0.32 | 29.6 ± 0.9 | 20.0 ± 1.7 | 27.82 ± 0.33 |
| Q-learning (λ=0) | 28.72 ± 0.34 | 40.0 ± 0.0 | 21.5 ± 1.7 | 27.92 ± 0.34 |

Paired difference in objective, Q-learning minus baseline (same seeded terms):

| Comparison | Difference |
| --- | --- |
| vs Never intervene | 1.460 ± 0.149 |
| vs Random | 0.757 ± 0.144 |
| vs Threshold rule | 0.099 ± 0.157 |

## 5. Fairness weight λ (Q-learning, one agent trained per λ)

| λ | Retained | Spent | Coverage gap (points) |
| --- | --- | --- | --- |
| 0 | 28.72 ± 0.34 | 40.0 ± 0.0 | 21.5 ± 1.7 |
| 2 | 28.59 ± 0.32 | 40.0 ± 0.0 | 20.7 ± 1.7 |
| 5 | 27.51 ± 0.32 | 40.0 ± 0.0 | 7.9 ± 0.8 |
| 10 | 27.27 ± 0.32 | 40.0 ± 0.0 | 7.6 ± 0.8 |
| 20 | 26.97 ± 0.32 | 40.0 ± 0.0 | 4.0 ± 0.7 |
