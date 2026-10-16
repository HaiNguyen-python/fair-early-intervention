# Learning guide (for me, Hai)

Goal: explain every part of this repo in an interview, connect it to my thesis, and defend every
number in the README. The thesis used Q-learning, DQN and PPO on OULAD; this repo rebuilds the
idea with production habits (tests, typed API, CI) and adds the risk model and fairness audit
that employers ask about first.

Start with [docs/how-it-works.md](docs/how-it-works.md) (the code walkthrough) and
[docs/decisions.md](docs/decisions.md) (the reasons). This file is the practice plan.

## 1. Run it (1 hour)

```bash
make install && make check
fair-intervention generate --out data/raw     # look at the CSVs: same columns as OULAD
fair-intervention report                      # ~3 min: docs/results.md + 2 charts
fair-intervention train && fair-intervention serve    # open http://localhost:8000/docs
```

Then: open `docs/results.md` and find, for each number in the README, the row it comes from.
Change `shock_rate_low_imd` in `data/synthetic.py` to 0.06 (same as everyone) and re-run the
report. The IMD gap should almost disappear. If it does not, find out why before moving on.

## 2. Read the code in this order

| File | What to understand | Question to answer yourself |
| --- | --- | --- |
| `data/synthetic.py` | Hidden engagement and ability, the planted shock, retakers | Why does the shock land after day 45? |
| `data/oulad.py` | Real file quirks (`?`, `10-20`) | What breaks if `?` is read as a string? |
| `features.py` | Cut-off day, missed vs late work, students who already left | Why is a late submission after the cut-off counted as "missed"? |
| `split.py` | Presentation order, retaker removal | What would leak if a retaker stayed in training? |
| `metrics.py` | `flag_top_k`, `recall_at_capacity`, ECE | Why can recall@20% never reach 100% here? |
| `risk.py` | Pipelines, Platt scaling, reasons | Why does calibration not change recall@capacity? |
| `fairness.py` | Group metrics, gap, group thresholds, score shift | Why compare before/after at the same list size? |
| `env.py` | Observation, budget rule, reveal week, shaped reward | Why did the plain −1 dropout reward fail? |
| `agents.py` | Q-update, ε schedule, threshold rule | What does each axis of the Q-table mean? |
| `policy_eval.py` | Common random numbers, paired intervals | Why is the paired interval narrower? |
| `serve.py` | Dependencies, request → simulator state | What happens if no Q-table exists? |

## 3. Concepts in my own words (fill in)

- **Data leakage in time:** ...
- **Temporal validation vs random split:** ...
- **PR-AUC vs ROC-AUC with 40% positives:** ...
- **Recall at capacity:** ...
- **Calibration and ECE:** ...
- **Fairness through unawareness, and why it failed here:** ...
- **Equal opportunity vs demographic parity:** ...
- **Group thresholds and their legal/ethical cost:** ...
- **MDP, Q-learning update, ε-greedy:** ...
- **Potential-based reward shaping:** ...
- **Common random numbers and paired confidence intervals:** ...
- **Off-policy evaluation (IPS, doubly robust) and why it is the next step:** ...

## 4. Interview questions this repo answers

1. *How do you build an early-warning model without leakage?* Cut-off day per week, the
   tampering test, temporal split, retakers removed.
2. *Which metric would you report to a school?* Recall and precision at the contact capacity,
   with the maximum possible recall next to it.
3. *How do you check a model for bias?* Recall, FPR and selection rate by group, with intervals,
   and only call a gap real when the group is big enough. Show the disability example.
4. *How do you fix it, and what does it cost?* Group thresholds: low-IMD recall 30% → 38%, gap
   15 → 3 points, other groups lose about 5 points of recall at the same list size. Explain why
   precision did not drop here and why that will not always hold.
5. *Why reinforcement learning, and is it worth it?* Budget and timing make it a sequential
   problem. Be honest: in this simulator Q-learning beats random and doing nothing, but not
   significantly the simple threshold rule. The λ curve is the more useful output.
6. *How would you evaluate a policy on real data?* Log interventions with their probabilities,
   then IPS or doubly robust off-policy evaluation, then a small randomised pilot.
7. *How does this relate to your thesis?* Same problem and dataset schema; this version adds a
   risk model, a fairness audit, shaping, paired evaluation and serving, and drops the deep RL
   to an optional extra.

## 5. Make it mine (do at least three)

- [ ] Run the report on real OULAD data and update the README tables and charts.
- [ ] Add reweighting as a second mitigation and compare it in the same table.
- [ ] Add the IMD group as a model feature and compare with group thresholds.
- [ ] Install `.[deep]`, run `fair-intervention deep --algo ppo`, and add the row to the results.
- [ ] Change the email effect to 5% and see whether the policy ranking changes.
- [ ] Read the `oulad-lakehouse` feature mart with `load_feature_mart` and train on it.

## 6. Quality checks I should be able to explain

- `make check` = ruff, ruff format, mypy, pytest with an 80% coverage gate (currently 98%).
- Why `deep.py` is excluded from coverage (its test is skipped without torch).
- Why the mitigation test uses the full default dataset (group recall needs a few hundred
  positives per group).

## 7. Honesty checklist before pushing

- [ ] Every number in the README is from `docs/results.md`, produced by my own run.
- [ ] The README says "synthetic" next to every result.
- [ ] I say clearly that the planted effect is invented and that simulator effects are
      assumptions.
- [ ] I never quote thesis numbers as results of this repo.
- [ ] I can explain why Q-learning vs the threshold rule is "not significant".

Note: the Docker image and the optional `[deep]` extra were not run where this repo was first
built (no Docker daemon, no torch). Build the image and run the PPO baseline once yourself.
