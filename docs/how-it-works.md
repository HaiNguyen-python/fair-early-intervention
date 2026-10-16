# How it works

This page follows one student from raw OULAD tables to a risk score, a fairness check and a
recommended action. File references point to `src/fair_intervention/`.

## 1. Data: the OULAD schema, real or synthetic

The Open University Learning Analytics Dataset has one row per student and presentation in
`studentInfo` (demographics and `final_result`), daily click totals in `studentVle`, marks in
`studentAssessment`, the coursework calendar in `assessments`, and withdrawal dates in
`studentRegistration`. Days count from the module start and can be negative.

- `data/oulad.py → load` reads the real CSVs, checks the columns, treats `?` as missing and fixes
  the `10-20` IMD spelling.
- `data/synthetic.py → generate` writes the same tables from a seeded random model. Each student
  has a hidden engagement and ability. Engagement drives clicks and on-time work; both lower the
  chance of failing. Retakers reuse their `id_student` in a later presentation.

**The planted problem.** Students from the most deprived areas (IMD 0-30%) are far more likely to
hit an outside shock (45% vs 6%) somewhere between day 45 and day 140. The shock causes most of
them to fail or withdraw, but it happens after the early-warning window, so early clicks look
normal. Gender, age band and disability have no planted effect.

## 2. Features as of week N, with no future leakage

`features.py → build_weekly_features(tables, week)` uses cut-off day `7 * week`:

| Feature | Meaning |
| --- | --- |
| `clicks_total`, `active_days`, `distinct_sites` | VLE activity before the cut-off |
| `clicks_last_week`, `clicks_trend` | last 7 days, and log change vs the 7 days before |
| `days_since_last_click` | silence; never clicked = cut-off + 1 |
| `clicks_before_start` | activity before day 0, a known early signal in OULAD |
| `n_submitted`, `n_missed`, `n_late`, `mean_score` | coursework due before the cut-off; work submitted after the cut-off counts as missed, because nobody knows yet |
| `num_of_prev_attempts`, `studied_credits` | known at registration |

Students who unregistered before the cut-off are dropped: they have left, so nobody can be
contacted. The exam is excluded. Protected attributes (`gender`, `age_band`, `imd_group`,
`disability`) are kept as columns for the fairness report but are not in `FEATURES`.

`tests/test_data_and_features.py → test_no_leakage_features_ignore_everything_after_the_cutoff`
multiplies every future click by 100, zeroes future scores and moves future withdrawals, then
checks that no feature changes. A second test flips every label and checks that only the target
moves.

## 3. Split: past cohorts train, the newest cohort tests

`split.py → temporal_split` orders presentations (2013B < 2013J < 2014B < 2014J): train on the
first two, validate on 2014B, test on 2014J. Any student in the test cohort is removed from
validation and training, and any validation student from training, so a retaker is never seen
twice.

## 4. Risk models and calibration

`risk.py → fit_risk_model` fits one of two models on the training cohorts:

- **Logistic regression** in a pipeline: median imputation with missing-value indicators
  (`mean_score` is missing for students with no marked work), scaling, `LogisticRegression`.
- **HistGradientBoostingClassifier** with shallow, regularised trees; it handles NaN natively.

Both are then calibrated on the validation cohort with `PlattCalibrator` (a logistic regression on
the logit of the raw score). It is monotone, so the ranking does not change.

`risk.py → evaluate` and `metrics.py` compute on the test cohort:

- **PR-AUC** (`average_precision_score`),
- **recall@capacity**: flag exactly `ceil(0.2 * n)` students with the highest risk
  (`flag_top_k`, stable sort so ties never overshoot the list size) and count how many of the
  real at-risk students are on the list,
- **ECE**: weighted mean gap between predicted and observed rates over ten bins.

`RiskModel.reasons` gives the top three features that push a student's log-odds up
(coefficient × standardised value), which is exact for the logistic model.

## 5. Fairness: measure, then mitigate

`fairness.py → group_metrics` computes per group: size, positives, base rate, selection rate,
recall with a 95% interval, FPR and precision. `equal_opportunity_gap` is the largest minus the
smallest recall among groups with at least 50 positives.

The report finds the planted problem: the low-IMD group's recall is well below the other groups,
while the gaps for gender and age stay within noise.

**Mitigation** (`fit_group_thresholds`): for a target recall `r`, each group's threshold is the
score of its `r`-th best positive on validation. The largest `r` whose combined list still fits
the 20% capacity wins. Groups with fewer than 30 validation positives keep the global threshold.
On the test cohort, `adjusted_scores` shifts each group's scores so that its threshold lines up
with the global one, and the top 20% are flagged. Both lists have exactly the same size, so the
before/after table compares like with like.

## 6. The intervention simulator

`env.py → InterventionEnv` is a Gymnasium environment for one 8-week term of 40 students and a
budget of 40 (email costs 1, tutor call 5).

1. `reset(seed)` draws each student's observed risk bucket (0-4), group (40% low-IMD), and a
   hidden extra risk (3× hazard) for 30% of low-IMD and 8% of other students. Dropout noise for
   every student and week is drawn up front, so every policy faces the same luck.
2. Each week the agent sees the enrolled students one at a time:
   `[bucket, week, budget bin, group, previous contacts]`. From week 3 the hidden risk shows:
   the observed bucket jumps by two.
3. `step(action)`: an unaffordable action becomes "no action", so the budget never goes negative.
   An email cuts the student's hazard by 20%, a call by 60%, and each repeat contact works half
   as well. Then the student drops out with probability equal to the hazard.
4. **Reward** (potential-based shaping): change in the student's expected survival to the end of
   term, minus `0.02 × cost`, minus `λ × (change in coverage gap)`. The coverage gap is the
   difference between groups in the share of truly at-risk students contacted at least once.
   Over a term, rewards add up to `retained − 0.02 × spend − λ × gap` minus a constant;
   `tests/test_policy.py` checks this identity.

## 7. Policies and evaluation

`agents.py`:

- `NeverPolicy`, `RandomPolicy`, and `ThresholdPolicy` (call each top-bucket student once, email
  each second-bucket student once).
- `QLearningAgent`: a table of shape `5 × 8 × 4 × 2 × 3 × 3`, ε-greedy with ε falling from 1 to
  0.05 over 70% of training, α = 0.05, γ = 0.995, 3,000 training terms.

`policy_eval.py` runs 300 evaluation terms (seeds from 100,000, never used in training) for each
policy and reports means with 95% intervals. `paired_difference` compares two policies term by
term, which removes most of the cohort-to-cohort noise. `lambda_sweep` trains one agent per
fairness weight and evaluates each one on the same terms.

`deep.py` (optional `[deep]` extra) trains DQN or PPO from stable-baselines3 on the same
environment for comparison; the test for it is skipped when the package is not installed.

## 8. Serving

`serve.py` is a FastAPI app.

```
POST /score {"week": 6, "features": {...}, "imd_band": "10-20%", "budget_left": 22}
```

1. The week's calibrated logistic model gives `risk`; `on_contact_list` compares it with the
   threshold that fills 20% of the validation cohort.
2. `top_reasons` lists the features that push the risk up.
3. `policy_observation` maps the request to the simulator's state (risk bucket edges
   0.2/0.4/0.6/0.8, course week 4 = simulator week 0, budget bin, group from the IMD band,
   previous contacts) and the Q-learning table picks the action. If no table has been trained,
   the threshold rule is used and the response says so in `policy`.

Model and policy are loaded through FastAPI dependencies (`get_bundle`, `get_policy`), so tests
replace them with a model trained on the test fixture.

## 9. Report

`report.py → build` runs everything above and `write` produces `docs/results.md` and the two
charts. Chart titles are computed from the results, so they stay true after a re-run on real
data.
