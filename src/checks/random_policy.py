"""Luck distribution: random long/short policies matched to PPO's typical activity.

Matching uses OOS exposure/turnover only for a retrospective diagnostic, never
test returns. It is not a deployable or independently calibrated strategy.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtest import account, metrics


def random_positions(n: int, amplitude: float, switch_probability: float, seed: int) -> np.ndarray:
    """Symmetric long/short exposure with a tunable probability of changing sign."""
    if n < 1 or not 0 <= amplitude <= 1 or not 0 <= switch_probability <= 1:
        raise ValueError("Invalid random-policy parameters")
    rng = np.random.default_rng(seed)
    changes = np.where(rng.random(n) < switch_probability, -1, 1)
    changes[0] = rng.choice([-1, 1])
    return amplitude * np.cumprod(changes)


def run(source: Path, output: Path, runs: int = 1000, annualization: int = 252) -> pd.DataFrame:
    """Save the entire luck distribution and observed activity matching errors."""
    if runs < 2:
        raise ValueError("Use at least two random runs")
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source / "metrics_by_seed.csv")
    rows, summaries, matching = [], [], []
    for asset_number, (asset, agents) in enumerate(raw[raw.strategy == "ppo"].groupby("asset")):
        frames = [pd.read_csv(source / f"{asset}_ppo_{seed}_oos.csv", parse_dates=["decision_time", "outcome_end"]).set_index("decision_time")
                  for seed in agents.seed]
        first = frames[0]
        amplitude = float(np.mean([f.position.abs().mean() for f in frames]))
        internal_turnover = float(np.mean([np.abs(np.diff(f.position)).mean() for f in frames]))
        switch = float(np.clip(internal_turnover / (2 * amplitude), 0, 1)) if amplitude else 0.0
        matching.append({"asset": asset, "target_mean_abs_exposure": amplitude,
                         "target_internal_turnover": internal_turnover, "switch_probability": switch,
                         "expected_internal_turnover": 2 * amplitude * switch,
                         "reference": str(source), "retrospective_matching": True})
        local = []
        for number in range(runs):
            seed = 100000 + 10000 * asset_number + number
            positions = random_positions(len(first), amplitude, switch, seed)
            frame = account(positions, first.asset_return, first.outcome_end, 0.0007, liquidate=True)
            record = {"asset": asset, "strategy": "random_matched", "seed": seed,
                      "mean_abs_exposure": float(np.abs(positions).mean()),
                      "internal_turnover": float(np.abs(np.diff(positions)).mean()),
                      **metrics(frame, annualization)}
            rows.append(record)
            local.append(record)
        distribution = pd.DataFrame(local)
        for agent in agents.itertuples():
            summaries.append({"asset": asset, "ppo_seed": agent.seed,
                              "ppo_sharpe": agent.sharpe,
                              "random_sharpe_percentile": float(100 * (distribution.sharpe <= agent.sharpe).mean()),
                              "random_sharpe_median": distribution.sharpe.median(),
                              "random_sharpe_p05": distribution.sharpe.quantile(.05),
                              "random_sharpe_p95": distribution.sharpe.quantile(.95),
                              "random_mean_turnover": distribution.turnover.mean(), "runs": runs})
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(distribution.sharpe, bins=40, alpha=.7, label=f"{runs} matched random policies")
        for agent in agents.itertuples():
            ax.axvline(agent.sharpe, linewidth=1, label=f"PPO seed {agent.seed}")
        ax.set(title=f"{asset}: how wide is luck?", xlabel="Net Sharpe", ylabel="Number of runs")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{asset}_random_policy.png", dpi=150)
        plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(output / "random_policy_runs.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "random_policy_comparison.csv", index=False)
    pd.DataFrame(matching).to_csv(output / "random_policy_matching.csv", index=False)
    return table
