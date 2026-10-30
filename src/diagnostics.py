"""Read existing PPO decisions and separate trading activity from cost drag."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import metrics
from .checks.cost_sensitivity import reprice


def position_statistics(frame: pd.DataFrame, annualization: int = 252) -> dict[str, float]:
    """Gross/net geometric growth and exposure fractions, without fitting anything."""
    position = frame.position
    threshold = 1e-8
    return {
        "gross_annualized_return": metrics(reprice(frame, 0), annualization)["annualized_return"],
        "net_annualized_return": metrics(frame, annualization)["annualized_return"],
        "daily_turnover": float(frame.turnover.mean()),
        "average_absolute_position": float(position.abs().mean()),
        "long_fraction": float((position > threshold).mean()),
        "short_fraction": float((position < -threshold).mean()),
        "flat_fraction": float((position.abs() <= threshold).mean()),
    }


def run(source: Path = Path("results/final"), root: Path = Path("results/diagnostics")) -> Path:
    """Create a new numbered diagnostic; retain every previous diagnostic."""
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while (root / f"run_{number:03}").exists():
        number += 1
    output = root / f"run_{number:03}"
    output.mkdir()
    raw = pd.read_csv(source / "metrics_by_seed.csv")
    rows = []
    for row in raw[raw.strategy == "ppo"].itertuples():
        frame = pd.read_csv(source / f"{row.asset}_ppo_{row.seed}_oos.csv")
        rows.append({"asset": row.asset, "seed": row.seed, "observations": len(frame),
                     **position_statistics(frame)})
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output / "metrics_by_seed.csv", index=False)
    names = list(position_statistics(frame))
    summary = by_seed.groupby("asset")[names].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.to_csv(output / "metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    for offset, (name, label) in enumerate([("gross", "Before costs"), ("net", "After costs")]):
        ax.bar(x + (offset-.5)*.35, summary[f"{name}_annualized_return_mean"] * 100,
               yerr=summary[f"{name}_annualized_return_std"] * 100, width=.35,
               label=label, capsize=4)
    ax.axhline(0, color="gray", linewidth=.8)
    ax.set_xticks(x, summary.index)
    ax.set(ylabel="Yearly compounded return (%)", title="Original PPO: mean ± spread across five starting points")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "before_after_costs.png", dpi=150)
    plt.close(fig)
    entries = []
    for asset, r in summary.iterrows():
        entries.append("| " + " | ".join([asset,
            f"{r.gross_annualized_return_mean:.2%} ± {r.gross_annualized_return_std:.2%}",
            f"{r.net_annualized_return_mean:.2%} ± {r.net_annualized_return_std:.2%}",
            f"{r.daily_turnover_mean:.3f}", f"{r.average_absolute_position_mean:.3f}",
            f"{r.long_fraction_mean:.1%}", f"{r.short_fraction_mean:.1%}", f"{r.flat_fraction_mean:.1%}"]) + " |")
    gross = "; ".join(f"{a} {r.gross_annualized_return_mean:.2%}" for a, r in summary.iterrows())
    net = "; ".join(f"{a} {r.net_annualized_return_mean:.2%}" for a, r in summary.iterrows())
    turnover = "; ".join(f"{a} {r.daily_turnover_mean:.3f}" for a, r in summary.iterrows())
    sentences = [f"Before costs, PPO's average yearly compounded returns were {gross}.",
                 f"After costs, they were {net}.",
                 f"Average daily turnover was {turnover}, so the agent changed roughly half an account's target exposure each day.",
                 "A positive before-cost return alone does not establish a prediction signal, meaning information that reliably helps choose profitable future trades, because market direction and luck also affect returns.",
                 "Trading expenses explain a substantial part of the loss, but weak before-cost performance relative to buy-and-hold also matters, so trading less alone is not proven to solve the problem."]
    report = """# Why did original PPO lose?

**Annualized return** means compounded growth expressed per year; **gross** is
before costs and **net** is after costs. **Turnover** is the absolute change in
target position: 0.5 means changing half an account's target exposure per day.
**Average absolute position** measures exposure size, ignoring direction. **Long**
benefits from rising prices, **short** benefits from falling prices, and **flat**
means a position within 0.00000001 of zero. Values are averages across five seeds
(repeatable random starting points); ± is their sample standard deviation, a
measure of variation. All six original performance metrics remain in `results/final/`.
Entry, direction reversal, fold carry, and final exit costs are included.

| Asset | Before costs, yearly | After costs, yearly | Daily turnover | Average absolute position | Long | Short | Flat |
| --- | --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(entries) + "\n\n## Five-sentence explanation\n\n" + "\n".join(sentences)
    (output / "report.md").write_text(report + "\n")
    (output / "manifest.json").write_text(json.dumps({"source": source.as_posix(), "fit_models": False,
                                                     "flat_threshold": 1e-8, "seeds": sorted(by_seed.seed.unique().tolist())}, indent=2))
    summary_path = Path("docs/RESULTS_SUMMARY.md")
    with summary_path.open("a") as stream:
        stream.write(f"\n\n## Original PPO diagnostic: {output.name}\n\n" + report.split("\n", 1)[1] +
                     f"\n\n![Before and after costs](../{output.as_posix()}/before_after_costs.png)\n")
    print(report)
    return output


if __name__ == "__main__":
    run()
