"""Command line entry point: generate | train | report | serve | deep."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .data import oulad, synthetic
from .env import EnvConfig

log = logging.getLogger("fair_intervention")


def load_tables(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], str]:
    """Real OULAD CSVs from ``--data-dir``, or a seeded synthetic dataset built in memory."""
    if args.source == "oulad":
        return oulad.load(Path(args.data_dir)), "OULAD data"
    cfg = synthetic.SyntheticConfig(students_per_presentation=args.students, seed=args.seed)
    return synthetic.generate(cfg), "Synthetic OULAD-shaped data"


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = synthetic.SyntheticConfig(students_per_presentation=args.students, seed=args.seed)
    for path in oulad.save(synthetic.generate(cfg), Path(args.out)):
        log.info("wrote %s", path)


def cmd_train(args: argparse.Namespace) -> None:
    from .agents import QLearningAgent
    from .risk import fit_bundle
    from .split import horizon_splits

    tables, label = load_tables(args)
    bundle = fit_bundle(horizon_splits(tables), args.capacity, label)
    out = Path(args.artifacts)
    bundle.save(out / "risk_model.joblib")
    agent = QLearningAgent(replace(EnvConfig(), fairness_lambda=args.fairness_lambda))
    agent.train(args.episodes)
    agent.save(out / "q_table.npy")
    summary = {"artifacts": str(out), "thresholds": bundle.thresholds, "data": label}
    print(json.dumps(summary, indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    from . import report

    tables, label = load_tables(args)
    results = report.build(
        tables,
        label,
        capacity=args.capacity,
        train_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
    )
    print(report.write(results, Path(args.out)))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("fair_intervention.serve:app", host=args.host, port=args.port)


def cmd_deep(args: argparse.Namespace) -> None:
    from .deep import train_sb3
    from .policy_eval import evaluate_policy

    policy = train_sb3(args.algo, EnvConfig(), args.timesteps)
    print(json.dumps(evaluate_policy(policy, EnvConfig(), args.eval_episodes), indent=2))


def _data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=["synthetic", "oulad"], default="synthetic")
    p.add_argument("--data-dir", default="data/raw", help="folder with the OULAD CSV files")
    p.add_argument("--students", type=int, default=1000, help="synthetic: per presentation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capacity", type=float, default=0.2, help="share of students tutors reach")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="fair-intervention")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write synthetic OULAD-shaped CSV files")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--students", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="fit risk models and the Q-learning policy for the API")
    _data_args(p)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--fairness-lambda", type=float, default=5.0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("report", help="run all experiments, write docs/results.md and charts")
    _data_args(p)
    p.add_argument("--out", default="docs")
    p.add_argument("--episodes", type=int, default=3000, help="Q-learning training episodes")
    p.add_argument("--eval-episodes", type=int, default=300)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("serve", help="run the scoring API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("deep", help="optional: DQN/PPO baseline (needs the [deep] extra)")
    p.add_argument("--algo", choices=["dqn", "ppo"], default="ppo")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=300)
    p.set_defaults(func=cmd_deep)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
