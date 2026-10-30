"""Time one full-budget fold before authorizing the entire final experiment.

Usage: python -m src.checks.estimate_runtime
This saves an estimate only. It NEVER starts the complete experiment.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from ..backtest import walk_forward
from ..features import decision_frame
from ..train_rl import train_predict
from .common import load_config, load_data


def main() -> None:
    config = load_config()
    data = load_data(config)
    counts = {}
    benchmark_seconds = []
    # Two representative assets reduce noise in a single very short measurement.
    for number, (asset, bars) in enumerate(data.items()):
        x, y, ends = decision_frame(bars)
        folds = list(walk_forward(x.index, ends, config["train_years"], config["test_months"], config["observation_window"]))
        counts[asset] = len(folds)
        if number < 2:
            start = perf_counter()
            train_predict(x, y, folds[0], config, config["seeds"][0], 0.)
            benchmark_seconds.append(perf_counter() - start)
    policies = sum(counts.values()) * len(config["seeds"])
    seconds = max(benchmark_seconds) * policies
    estimate = {"config": config, "folds_per_asset": counts, "policy_fits": policies,
                "benchmark_seconds_per_policy": benchmark_seconds,
                "estimated_training_minutes": seconds / 60,
                "conservative_total_minutes": seconds * 1.3 / 60 + 2,
                "more_than_30_minutes": seconds * 1.3 / 60 + 2 > 30,
                "method": "Two full-budget fold benchmarks; slowest times all policies + 30% margin + 2 minutes IO/plots."}
    output = Path("results/checks/runtime_estimate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(estimate, indent=2))
    print(json.dumps(estimate, indent=2), flush=True)


if __name__ == "__main__":
    main()
