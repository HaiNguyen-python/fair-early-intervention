# Design decisions

Short records of the main choices: what was decided, why, and what it costs. Format inspired by
Architecture Decision Records (ADRs). Numbers refer to the synthetic run in
[results.md](results.md).

## 1. Predict at weeks 4, 6 and 8, not at the end of term

- **Decision:** one model per early-warning week, each built only from data before day `7 * N`.
- **Why:** a prediction is only useful while there is still time to act. A model that uses the
  whole term's clicks looks great and helps nobody.
- **Cost:** less signal. PR-AUC rises from week 4 to week 8 on synthetic data; the school has to
  choose between an earlier, weaker list and a later, stronger one.

## 2. Temporal split by presentation, grouped by student

- **Decision:** train on the oldest presentations, calibrate and tune thresholds on the next one,
  test on the newest. Any student who appears in a later split is removed from earlier ones.
- **Why:** that is how the model would be used (fit on past cohorts, score the current one).
  OULAD has retakers; without the grouping, the model could recognise a person instead of a
  pattern.
- **Cost:** fewer training rows, and a single test cohort, so test numbers carry sampling noise.
  Rolling-origin validation over several presentations would give error bars on the model
  metrics too.

## 3. Recall at capacity as the main metric

- **Decision:** report recall and precision for the top 20% of students, next to PR-AUC.
- **Why:** tutors have a fixed number of hours. "How many struggling students does a list of
  this size reach?" is the question they ask. The table also shows `max recall`, because when
  40% of students are at risk, a 20% list can reach at most half of them.
- **Cost:** the capacity (20%) is an assumption; a real deployment would ask the institution.

## 4. Logistic regression is served, not gradient boosting

- **Decision:** the API serves the calibrated logistic model; HistGradientBoosting is the
  comparison.
- **Why:** on this data the two are within about 1 point of PR-AUC, and logistic regression gives
  exact per-student reasons (coefficient × standardised value adds up to the log-odds). Staff are
  more willing to act on a flag they can understand.
- **Cost:** on real OULAD, trees may win by more. Then the reasons would need SHAP or similar,
  which adds a dependency and an approximation.
- **Alternative:** LightGBM. sklearn's HistGradientBoosting is the same algorithm family and
  keeps the dependency list short.

## 5. Platt calibration on the validation cohort

- **Decision:** a one-feature logistic regression on the logit of the raw score.
- **Why:** the API returns a probability that people will read as "6 in 10". Platt scaling is
  monotone, so ranking (and recall at capacity) does not change; it needs few rows.
- **Cost:** on this synthetic data both models were already well calibrated, so the gain in ECE is
  small. Isotonic regression is more flexible but needs more data and can overfit.

## 6. Protected attributes are not model inputs ("fairness through unawareness")

- **Decision:** gender, age band, IMD band and disability are kept for measurement only.
- **Why:** this is the common starting point, and the report shows its limit: the model never
  sees IMD, yet its recall for the low-IMD group is 15 points lower, because behaviour carries
  different information for different groups.
- **Cost:** unawareness does not prevent unfairness, so a measurement step is required.

## 7. Equal opportunity as the fairness criterion

- **Decision:** compare recall (TPR) across groups; also show FPR and selection rate.
- **Why:** the harm of a missed flag is a struggling student who gets no help. Equal recall asks
  that struggling students are found at the same rate whatever group they belong to.
- **Cost:** it ignores false positives (an unneeded email). Here the cost of a false positive is
  low, so this trade is acceptable; for punitive uses of a model it would not be.
- **Small groups:** gaps are only computed for groups with at least 50 positives. The disability
  gap (9 points with a ±10-point interval for 75 positives) is an example of a gap that is
  mostly noise; no effect was planted there.

## 8. Mitigation: group thresholds at a fixed list size

- **Decision:** choose per-group thresholds on validation so that recall is equal, keep the
  total list at 20%, and apply them to the test cohort by shifting scores and taking the top 20%.
- **Why:** it is simple, transparent, model-agnostic and directly targets equal recall. Keeping
  the list size fixed makes "before" and "after" comparable.
- **Cost:** the extra low-IMD flags come from other groups: high- and mid-IMD recall fall by
  about 5 points. Overall precision did not drop on this data, because the model under-ranks
  low-IMD students who really are at higher risk. Do not expect that on every dataset.
- **Policy question:** the mitigation needs the IMD band at decision time. Using a group label
  this way must be agreed with the institution (and checked against law; disability, for
  example, is special-category data under GDPR).
- **Alternatives:** reweighting during training; adding the group as a feature; collecting the
  missing signal (outside workload) instead. The last one fixes the cause.

## 9. A small tabular RL problem, not deep RL

- **Decision:** a Gymnasium environment with a discrete observation (risk bucket, week, budget
  bin, group, previous contacts) and tabular Q-learning; DQN/PPO only under the optional `[deep]`
  extra.
- **Why:** the state space has 960 cells. A table learns it in seconds, and every value can be
  inspected. Deep RL would add torch to CI for no gain at this size.
- **Cost:** the environment had to be designed to be learnable (see 10). Results do not transfer
  to large state spaces.
- **Extension to the brief:** "previous contacts" is added to the state; without it no policy can
  know that a second call works only half as well.

## 10. Potential-based reward shaping

- **Decision:** the per-step reward is the change in each student's expected survival to the end
  of term, minus cost, minus λ × the change in the coverage gap.
- **Why:** with a plain "−1 when a student drops out" reward, the effect of a call in week 1
  appears weeks later on a different student's turn; in an earlier version of this repo
  Q-learning trained that way lost to the threshold rule. Shaping with a potential (Ng et al.,
  1999) keeps the optimal policy and makes the effect visible at the step where the action was
  taken. A test checks that rewards add up to the objective minus a constant.
- **Cost:** the environment uses its true hazards to compute the shaping term. That is fine in a
  simulator but impossible on real data.

## 11. The simulator's effect sizes are assumptions

- **Decision:** document every number in `EnvConfig` (email −20% hazard, call −60%, halving per
  repeat contact, hidden risk 3×, revealed in week 3, budget 40 for 40 students).
- **Why:** the simulator compares policies under stated assumptions. It is not evidence about
  what an email does.
- **Cost:** the ranking of policies can change if the assumptions change. The honest next step is
  off-policy evaluation on logged interventions (inverse propensity scoring or doubly robust
  estimators) using records of who was actually contacted and what happened, which OULAD does
  not contain.

## 12. Evaluation with common random numbers and paired intervals

- **Decision:** every policy is evaluated on the same 300 seeded terms (same cohorts, same
  dropout noise); differences are reported as paired 95% intervals.
- **Why:** cohort-to-cohort variation is much larger than the gap between good policies. Paired
  differences remove most of it. Evaluation seeds (100,000+) never overlap training seeds.
- **Finding:** Q-learning clearly beats never intervening and random spending, but its edge over
  the hand-written threshold rule is not significant (the interval includes 0). That is reported,
  not hidden.

## 13. Synthetic data with a planted problem

- **Decision:** ship a seeded generator in the OULAD schema; plant one fairness problem on purpose
  (low-IMD students hit outside shocks after the early-warning window).
- **Why:** CI cannot download OULAD, and a planted effect makes it possible to check that the
  analysis finds what is there and does not find what is not (gender, age and disability have
  no planted effect).
- **Cost:** synthetic results show that the pipeline works, not how the model performs on real
  students.
