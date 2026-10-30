"""Reprice frozen OOS positions at 0, 5, 7 (default), and 20 bps all-in cost.

This isolates execution friction. Policies are NOT retrained at each cost.
All-in means fees plus slippage, with no extra hidden 2-bp slippage added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtest import metrics
from .common import summarize


def reprice(frame: pd.DataFrame, all_in_bps: float) -> pd.DataFrame:
    """Apply a new cost rate to identical turnover, including existing final exit."""
    if not np.isfinite(all_in_bps) or all_in_bps < 0:
        raise ValueError("Cost must be finite and nonnegative")
    result = frame.copy()
    result["cost"] = result.turnover * all_in_bps / 10000
    result["net_return"] = result.position * result.asset_return - result.cost
    return result


def run(source: Path, output: Path, annualization: int = 252) -> pd.DataFrame:
    """Save seed metrics, mean/std, and Sharpe-vs-cost plots for every asset."""
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source / "metrics_by_seed.csv")
    rows = []
    for item in raw.itertuples():
        frame = pd.read_csv(source / f"{item.asset}_{item.strategy}_{item.seed}_oos.csv")
        for cost in [0, 5, 7, 20]:
            rows.append({"asset": item.asset, "strategy": item.strategy, "seed": item.seed,
                         "all_in_cost_bps": cost, **metrics(reprice(frame, cost), annualization)})
    table = pd.DataFrame(rows)
    table.to_csv(output / "cost_sensitivity_by_seed.csv", index=False)
    summary = summarize(table, ["asset", "strategy", "all_in_cost_bps"])
    summary.to_csv(output / "cost_sensitivity.csv", index=False)
    for asset, group in summary.groupby("asset"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for strategy, values in group.groupby("strategy"):
            ax.errorbar(values.all_in_cost_bps, values.sharpe_mean,
                        yerr=values.sharpe_std.fillna(0), marker="o", capsize=3, label=strategy)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(title=f"{asset}: same positions, different costs (mean ± seed std)",
               xlabel="All-in one-way cost (bps; fees + slippage)", ylabel="Net Sharpe")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{asset}_cost_sensitivity.png", dpi=150)
        plt.close(fig)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/final"))
    parser.add_argument("--output", type=Path, default=Path("results/checks/cost_sensitivity"))
    args = parser.parse_args()
    run(args.source, args.output)
