"""Run the full five-seed experiment only after an explicit runtime estimate.

Usage: python -m src.checks.run_final
The --allow-long-run flag is for an explicitly accepted estimate over 30 minutes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import evaluate, load_config
from .cost_sensitivity import run as costs
from .random_policy import run as random
from .report import final_report


def validate_runtime(estimate: dict, config: dict, allow_long: bool) -> None:
    """Prevent silently starting a long experiment or using a stale estimate."""
    if estimate["config"] != config:
        raise ValueError("Settings changed since runtime estimate; rerun src.checks.estimate_runtime")
    if estimate["more_than_30_minutes"] and not allow_long:
        raise ValueError("Estimated runtime exceeds 30 minutes. Obtain the user's decision before --allow-long-run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-long-run", action="store_true")
    args = parser.parse_args()
    config = load_config()
    estimate_path = Path("results/checks/runtime_estimate.json")
    if not estimate_path.exists():
        raise ValueError("First run python -m src.checks.estimate_runtime and review its runtime estimate")
    estimate = json.loads(estimate_path.read_text())
    validate_runtime(estimate, config, args.allow_long_run)
    output = Path("results/final")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    evaluate(config, output)
    costs(output, output, config["annualization"])
    random(output, Path("results/checks/final_random_policy"), 1000, config["annualization"])
    final_report()


if __name__ == "__main__":
    main()
