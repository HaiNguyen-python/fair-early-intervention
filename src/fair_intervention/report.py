"""Run every experiment and write docs/results.md plus two charts.

    docs/results.md                  all tables quoted in the README
    docs/img/recall_by_group.png     recall per IMD group, before vs after mitigation
    docs/img/lambda_tradeoff.png     retention vs coverage gap as the fairness weight grows

Every number in the docs comes from this module, so re-running it on real OULAD data
(``fair-intervention report --source oulad --data-dir ...``) replaces them all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agents import QLearningAgent
from .env import EnvConfig
from .fairness import (
    adjusted_scores,
    equal_opportunity_gap,
    fit_group_thresholds,
    global_threshold,
    group_metrics,
)
from .features import PROTECTED, TARGET
from .metrics import flag_top_k, recall_at_capacity
from .policy_eval import (
    baselines,
    episode_runs,
    lambda_sweep,
    paired_difference,
    summarise,
)
from .risk import evaluate, fit_risk_model
from .split import Split, horizon_splits

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREY = "#2a78d6", "#b7b7b2"
GROUP_LABELS = {
    "high": "High IMD (70-100%)",
    "mid": "Mid IMD (30-70%)",
    "low": "Low IMD (0-30%)",
    "unknown": "IMD unknown",
}
MITIGATED = "imd_group"


@dataclass
class Results:
    data_label: str
    capacity: float
    fairness_week: int
    presentations: dict[str, list[str]]
    risk_rows: list[dict[str, Any]]
    fairness_tables: dict[str, pd.DataFrame]
    mitigation: dict[str, Any]
    env_config: EnvConfig
    eval_episodes: int
    policy_rows: list[dict[str, Any]]
    paired: dict[str, tuple[float, float]]
    sweep: list[dict[str, Any]] = field(default_factory=list)


def run_risk(splits: dict[int, Split], capacity: float) -> list[dict[str, Any]]:
    rows = []
    for week, split in splits.items():
        for kind in ("logistic", "hist_gb"):
            model = fit_risk_model(split.train, split.val, kind, week)
            rows.append({"week": week, "model": kind, **evaluate(model, split.test, capacity)})
    return rows


def run_fairness(
    split: Split, week: int, capacity: float
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Group metrics for the plain top-K list, then group thresholds for the IMD groups."""
    model = fit_risk_model(split.train, split.val, "logistic", week)
    p_val, p_test = model.predict_proba(split.val), model.predict_proba(split.test)
    y = split.test[TARGET].to_numpy()
    before = flag_top_k(p_test, capacity)
    tables = {a: group_metrics(y, before, split.test[a].to_numpy()) for a in PROTECTED}

    thresholds = fit_group_thresholds(
        p_val, split.val[TARGET].to_numpy(), split.val[MITIGATED].to_numpy(), capacity
    )
    shifted = adjusted_scores(
        p_test, split.test[MITIGATED].to_numpy(), thresholds, global_threshold(p_val, capacity)
    )
    after = flag_top_k(shifted, capacity)
    mitigation: dict[str, Any] = {"thresholds": thresholds}
    for name, flagged, score in (("before", before, p_test), ("after", after, shifted)):
        table = group_metrics(y, flagged, split.test[MITIGATED].to_numpy())
        overall = recall_at_capacity(y, score, capacity)
        mitigation[name] = {
            "table": table,
            "precision": overall["precision"],
            "recall": overall["recall"],
            "gap": equal_opportunity_gap(table),
        }
    return tables, mitigation


def run_policies(
    config: EnvConfig, train_episodes: int, eval_episodes: int, lambdas: Sequence[float]
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]], list[dict[str, Any]]]:
    agents: dict[float, QLearningAgent] = {}
    sweep = lambda_sweep(
        config,
        lambdas,
        train_episodes,
        eval_episodes,
        on_trained=lambda lam, agent: agents.__setitem__(lam, agent),
    )
    base_cfg = replace(config, fairness_lambda=0.0)
    runs = {p.name: episode_runs(p, base_cfg, eval_episodes) for p in baselines()}
    runs["Q-learning (λ=0)"] = episode_runs(agents[lambdas[0]], base_cfg, eval_episodes)
    rows = [{"policy": name, **summarise(r)} for name, r in runs.items()]
    q = runs["Q-learning (λ=0)"]
    paired = {
        "vs Never intervene": paired_difference(q, runs["Never intervene"], "objective"),
        "vs Random": paired_difference(q, runs["Random"], "objective"),
        "vs Threshold rule": paired_difference(q, runs["Threshold rule"], "objective"),
    }
    return rows, paired, sweep


def build(
    tables: dict[str, pd.DataFrame],
    data_label: str,
    capacity: float = 0.2,
    fairness_week: int = 6,
    env_config: EnvConfig | None = None,
    train_episodes: int = 3000,
    eval_episodes: int = 300,
    lambdas: Sequence[float] = (0.0, 2.0, 5.0, 10.0, 20.0),
) -> Results:
    env_config = env_config or EnvConfig()
    splits = horizon_splits(tables)
    risk_rows = run_risk(splits, capacity)
    fairness_tables, mitigation = run_fairness(splits[fairness_week], fairness_week, capacity)
    policy_rows, paired, sweep = run_policies(env_config, train_episodes, eval_episodes, lambdas)
    return Results(
        data_label=data_label,
        capacity=capacity,
        fairness_week=fairness_week,
        presentations=splits[fairness_week].presentations,
        risk_rows=risk_rows,
        fairness_tables=fairness_tables,
        mitigation=mitigation,
        env_config=env_config,
        eval_episodes=eval_episodes,
        policy_rows=policy_rows,
        paired=paired,
        sweep=sweep,
    )


# -- markdown -----------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _ci(value: tuple[float, float], fmt: str = ".2f") -> str:
    return f"{value[0]:{fmt}} ± {value[1]:{fmt}}"


def _pct(x: Any) -> str:
    return "n/a" if pd.isna(x) else f"{float(x):.1%}"


def results_markdown(res: Results) -> str:
    cap = f"{res.capacity:.0%}"
    p = res.presentations
    out = [
        "# Results",
        "",
        f"Data: **{res.data_label}**. Every number below is produced by `fair-intervention "
        "report`; nothing is copied from elsewhere.",
        "",
        f"Temporal split: train on {', '.join(p['train'])}, calibrate and choose thresholds on "
        f"{', '.join(p['val'])}, test on {', '.join(p['test'])}. Students who appear in a later "
        "split are removed from the earlier ones.",
        "",
        "## 1. Risk model (test cohort)",
        "",
        f"Contact capacity: tutors can reach the top {cap} of enrolled students. "
        "`max recall` is the best any model could do with a list that size.",
        "",
    ]
    rows = [
        [
            str(r["week"]),
            "Logistic regression" if r["model"] == "logistic" else "Hist. gradient boosting",
            f"{int(r['n'])}",
            _pct(r["base_rate"]),
            f"{r['pr_auc']:.3f}",
            _pct(r["recall_at_capacity"]),
            _pct(r["max_recall"]),
            _pct(r["precision_at_capacity"]),
            f"{r['ece_raw']:.3f} → {r['ece_calibrated']:.3f}",
        ]
        for r in res.risk_rows
    ]
    out.append(
        _table(
            [
                "Week",
                "Model",
                "Students",
                "At-risk rate",
                "PR-AUC",
                f"Recall@{cap}",
                "Max recall",
                f"Precision@{cap}",
                "ECE raw → calibrated",
            ],
            rows,
        )
    )
    out += [
        "",
        f"## 2. Fairness at week {res.fairness_week} (logistic model, top {cap} list)",
        "",
        "Recall = share of students who later fail or withdraw that the list reaches. "
        "± is a 95% interval. The equal-opportunity gap only counts groups with at least 50 "
        "positives.",
        "",
    ]
    for attr, table in res.fairness_tables.items():
        rows = [
            [
                str(t.group),
                str(t.n),
                str(t.positives),
                _pct(t.base_rate),
                _pct(t.selection_rate),
                f"{_pct(t.recall)} ± {t.recall_ci:.1%}",
                _pct(t.fpr),
            ]
            for t in table.itertuples()
        ]
        out += [
            f"**{attr}** (equal-opportunity gap: {equal_opportunity_gap(table) * 100:.1f} points)",
            "",
            _table(
                ["Group", "Students", "Positives", "Base rate", "Selected", "Recall", "FPR"], rows
            ),
            "",
        ]
    m = res.mitigation
    b, a = m["before"], m["after"]
    groups = list(b["table"]["group"])
    rows = [
        ["Overall precision", _pct(b["precision"]), _pct(a["precision"])],
        ["Overall recall", _pct(b["recall"]), _pct(a["recall"])],
        ["Equal-opportunity gap (IMD), points", f"{b['gap'] * 100:.1f}", f"{a['gap'] * 100:.1f}"],
    ]
    for g in groups:
        rb = b["table"].set_index("group").loc[g]
        ra = a["table"].set_index("group").loc[g]
        label = GROUP_LABELS.get(g, g)
        rows.append([f"Recall, {label}", _pct(rb.recall), _pct(ra.recall)])
        rows.append([f"FPR, {label}", _pct(rb.fpr), _pct(ra.fpr)])
    out += [
        "## 3. Mitigation: group thresholds for IMD groups",
        "",
        f"Thresholds chosen on the validation cohort, applied to the test cohort. Both lists "
        f"contact exactly {cap} of students, so precision is comparable.",
        "",
        _table(["Metric", "One global threshold", "Group thresholds"], rows),
        "",
        "## 4. Intervention policies (simulator)",
        "",
        f"{res.eval_episodes} evaluation terms of {res.env_config.n_students} students, "
        f"{res.env_config.n_weeks} weeks, budget {res.env_config.budget:.0f} "
        f"(email {res.env_config.email_cost:.0f}, call {res.env_config.call_cost:.0f}). "
        "Mean ± 95% interval. Objective = retained − "
        f"{res.env_config.cost_weight} × spend (λ = 0). The simulator's effect sizes are "
        "assumptions, see docs/decisions.md.",
        "",
    ]
    rows = [
        [
            r["policy"],
            _ci(r["retained"]),
            _ci(r["spent"], ".1f"),
            _ci((r["coverage_gap"][0] * 100, r["coverage_gap"][1] * 100), ".1f"),
            _ci(r["objective"]),
        ]
        for r in res.policy_rows
    ]
    out.append(_table(["Policy", "Retained", "Spent", "Coverage gap (points)", "Objective"], rows))
    out += [
        "",
        "Paired difference in objective, Q-learning minus baseline (same seeded terms):",
        "",
        _table(["Comparison", "Difference"], [[k, _ci(v, ".3f")] for k, v in res.paired.items()]),
        "",
        "## 5. Fairness weight λ (Q-learning, one agent trained per λ)",
        "",
    ]
    rows = [
        [
            f"{r['lambda']:g}",
            _ci(r["retained"]),
            _ci(r["spent"], ".1f"),
            _ci((r["coverage_gap"][0] * 100, r["coverage_gap"][1] * 100), ".1f"),
        ]
        for r in res.sweep
    ]
    out.append(_table(["λ", "Retained", "Spent", "Coverage gap (points)"], rows))
    return "\n".join(out) + "\n"


# -- charts -------------------------------------------------------------------------------
def _style(ax: Any, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)


def plot_recall_by_group(res: Results, path: Path) -> None:
    import matplotlib.pyplot as plt

    before = res.mitigation["before"]["table"].set_index("group")
    after = res.mitigation["after"]["table"].set_index("group")
    groups = [g for g in GROUP_LABELS if g in before.index][::-1]
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    for i, g in enumerate(groups):
        b, a = before.loc[g, "recall"] * 100, after.loc[g, "recall"] * 100
        ax.plot([b, a], [i, i], color=GREY, linewidth=2, zorder=1)
        ax.scatter([b], [i], s=70, color=GREY, zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.scatter([a], [i], s=70, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
        left_is_before = b <= a
        for value, is_before in ((b, True), (a, False)):
            on_left = is_before == left_is_before
            ax.annotate(
                f"{'before' if is_before else 'after'} {value:.0f}%",
                (value, i),
                xytext=(-10 if on_left else 10, 0),
                textcoords="offset points",
                ha="right" if on_left else "left",
                va="center",
                fontsize=9,
                color=INK_2 if is_before else INK,
            )
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([GROUP_LABELS[g] for g in groups], color=INK, fontsize=9.5)
    ax.set_xlim(15, 65)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    low_b, low_a = before.loc["low", "recall"], after.loc["low", "recall"]
    _style(
        ax,
        f"Group thresholds raise low-IMD recall from {low_b:.0%} to {low_a:.0%} "
        f"with the same {res.capacity:.0%} contact list",
        f"Recall among week-{res.fairness_week} test students who later fail or withdraw. "
        f"Grey: one global threshold. Blue: group thresholds. {res.data_label}.",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_tradeoff(res: Results, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    gaps = np.array([r["coverage_gap"][0] * 100 for r in res.sweep])
    kept = np.array([r["retained"][0] for r in res.sweep])
    ax.errorbar(
        gaps,
        kept,
        xerr=[r["coverage_gap"][1] * 100 for r in res.sweep],
        yerr=[r["retained"][1] for r in res.sweep],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        elinewidth=1,
        ecolor=GRID,
        capsize=0,
        zorder=3,
    )
    # The line rises to the right, so labels go above-left or below-right of it, alternating.
    for k, (r, x, y) in enumerate(zip(res.sweep, gaps, kept, strict=True)):
        above = k % 2 == 0
        ax.annotate(
            ("Q-learning, " if k == 0 else "") + f"λ={r['lambda']:g}",
            (x, y),
            xytext=(-10, 8) if above else (10, -14),
            textcoords="offset points",
            ha="right" if above else "left",
            fontsize=9,
            color=INK,
        )
    for r in res.policy_rows:
        if r["policy"].startswith("Q-learning"):
            continue
        x, y = r["coverage_gap"][0] * 100, r["retained"][0]
        ax.scatter([x], [y], s=60, color=GREY, zorder=2, edgecolor=SURFACE, linewidth=2)
        edge = x < 1  # at the y-axis there is no room to centre the label
        ax.annotate(
            r["policy"],
            (x, y),
            xytext=(8, -12) if edge else (0, -18),
            textcoords="offset points",
            ha="left" if edge else "center",
            fontsize=9,
            color=INK_2,
        )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("Coverage gap among truly at-risk students, points", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Students retained (of {res.env_config.n_students})", color=INK_2, fontsize=9)
    ax.set_xlim(left=-2, right=max(gaps.max(), 25) + 8)
    first, last = res.sweep[0], res.sweep[-1]
    _style(
        ax,
        f"Raising λ from {first['lambda']:g} to {last['lambda']:g} cuts the coverage gap from "
        f"{first['coverage_gap'][0] * 100:.0f} to {last['coverage_gap'][0] * 100:.0f} points "
        f"for {first['retained'][0] - last['retained'][0]:.1f} fewer retained students",
        f"Mean of {res.eval_episodes} simulated terms with 95% intervals. Effect sizes are "
        "assumptions. Synthetic simulator.",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write(res: Results, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = results_markdown(res)
    (out_dir / "results.md").write_text(text)
    plot_recall_by_group(res, out_dir / "img" / "recall_by_group.png")
    plot_tradeoff(res, out_dir / "img" / "lambda_tradeoff.png")
    return text
