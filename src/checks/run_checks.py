"""Run fixed-budget label-shuffle checks, cost sensitivity, and a luck comparison.

Usage: python -m src.checks.run_checks
Default: all assets/folds, five seeds, 1024 PPO steps for each paired experiment.
This is a diagnostic budget, separate from the full 10000-step final experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import evaluate, load_config
from .cost_sensitivity import run as costs
from .random_policy import run as random
from .report import checks_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--random-runs", type=int, default=1000)
    args = parser.parse_args()
    config = load_config(args.config)
    config["ppo_timesteps"] = args.timesteps
    root = Path("results/checks")
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps({"config": config, "random_runs": args.random_runs,
        "costs_all_in_bps": [0, 5, 7, 20], "test_tuning": False,
        "notes": "All folds; paired ordinary/shuffled training. Frozen-position cost sensitivity. Retrospectively activity-matched random policies."}, indent=2))
    evaluate(config, root / "ordinary")
    evaluate(config, root / "shuffled", shuffle=True)
    costs(root / "ordinary", root / "cost_sensitivity", config["annualization"])
    random(root / "ordinary", root / "random_policy", args.random_runs, config["annualization"])
    checks_report(root)


if __name__ == "__main__":
    main()
