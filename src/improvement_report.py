"""Describe every completed variant without choosing new parameters or hiding failures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import equity_and_drawdown
from .checks.report import markdown_table
from .round_variants import VARIANTS

LABELS = {
    "buy_hold": "Buy-and-hold", "ma_50_200": "50/200 average rule",
    "gradient_boosting": "Original forecast", "ppo": "Original PPO",
    "ppo_turnover_penalty": "PPO + turnover penalty", "ppo_long_only": "PPO long-only",
    "vol_target_buy_hold": "Volatility-targeted holding", "gb_fixed_features": "Fixed-feature forecast",
}


def comparison_table(table: pd.DataFrame, *, full: bool = True) -> str:
    """Report all strategies; mean/std are taken over individual seed metrics."""
    columns = [("annualized_return", "Yearly net return", True),
               ("annualized_volatility", "Yearly volatility", True),
               ("sharpe", "Sharpe", False), ("max_drawdown", "Worst drawdown", True),
               ("turnover", "Daily turnover", False), ("hit_rate", "Hit rate", True)]
    if not full:
        columns = [columns[0], columns[2]]
    rows = []
    for row in table.itertuples():
        cells = [row.asset, LABELS[row.strategy]]
        for name, _, percentage in columns:
            mean, std = getattr(row, name+"_mean"), getattr(row, name+"_std")
            cells.append(f"{mean:.2%} ± {std:.2%}" if percentage else f"{mean:.3f} ± {std:.3f}")
        rows.append(cells)
    return markdown_table(["Asset", "Strategy"] + [label for _, label, _ in columns], rows)


def make_plots(output: Path, raw: pd.DataFrame) -> None:
    """Display mean seed paths and full seed ranges; these are not traded ensembles."""
    for asset, rows in raw.groupby("asset"):
        paths = {}
        for strategy in LABELS:
            subset = rows[rows.strategy == strategy]
            frames = [pd.read_csv(output / f"{asset}_{strategy}_{seed}_oos.csv", parse_dates=["decision_time", "outcome_end"])
                      for seed in subset.seed]
            dates = pd.DatetimeIndex(frames[0].outcome_end)
            equity, drawdown = [], []
            for frame in frames:
                e, d = equity_and_drawdown(frame.net_return)
                equity.append(e.to_numpy())
                drawdown.append(d.to_numpy())
            paths[strategy] = (dates, np.asarray(equity), np.asarray(drawdown))
        for kind, field in [("equity", 1), ("drawdown", 2)]:
            fig, ax = plt.subplots(figsize=(12, 6))
            for strategy, values in paths.items():
                dates, matrix = values[0], values[field]
                line, = ax.plot(dates, matrix.mean(axis=0), label=LABELS[strategy], linewidth=1.2)
                ax.fill_between(dates, matrix.min(axis=0), matrix.max(axis=0), color=line.get_color(), alpha=.07)
            ax.set(title=f"{asset}: {kind} after costs; mean and range across five starting points",
                   xlabel="Outcome date", ylabel="Account value (start = 1)" if kind == "equity" else "Fall from prior peak")
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=.15)
            fig.tight_layout()
            fig.savefig(output / f"{asset}_{kind}.png", dpi=150)
            plt.close(fig)


def write_report(output: Path) -> dict:
    """Generate a complete table and eight sentences from the frozen round's outcomes."""
    metadata = json.loads((output / "manifest.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("An incomplete/failed round cannot be described as completed")
    summary = pd.read_csv(output / "metrics.csv")
    raw = pd.read_csv(output / "metrics_by_seed.csv")
    exposure = pd.read_csv(output / "exposure_by_seed.csv").groupby(["asset", "strategy"]).mean(numeric_only=True)
    assets = sorted(summary.asset.unique())
    return_wins, both_wins = [], []
    penalty_less_trading = penalty_better_return = long_better_return = vol_lower_risk = forecast_better_return = 0
    for asset in assets:
        group = summary[summary.asset == asset].set_index("strategy")
        buy = group.loc["buy_hold"]
        for variant in VARIANTS:
            row = group.loc[variant]
            if row.annualized_return_mean > buy.annualized_return_mean:
                return_wins.append({"asset": asset, "variant": variant})
                if row.sharpe_mean > buy.sharpe_mean:
                    both_wins.append({"asset": asset, "variant": variant})
        penalty_less_trading += int(exposure.loc[(asset, "ppo_turnover_penalty"), "daily_turnover"] < exposure.loc[(asset, "ppo"), "daily_turnover"])
        penalty_better_return += int(group.loc["ppo_turnover_penalty", "annualized_return_mean"] > group.loc["ppo", "annualized_return_mean"])
        long_better_return += int(group.loc["ppo_long_only", "annualized_return_mean"] > group.loc["ppo", "annualized_return_mean"])
        vol_lower_risk += int(group.loc["vol_target_buy_hold", "annualized_volatility_mean"] < buy.annualized_volatility_mean)
        forecast_better_return += int(group.loc["gb_fixed_features", "annualized_return_mean"] > group.loc["gradient_boosting", "annualized_return_mean"])
    if not return_wins:
        first = "None of the four new variants beat buy-and-hold's average yearly return after costs on any asset."
    else:
        descriptions = "; ".join(f"{win['asset']}: {LABELS[win['variant']]}" for win in return_wins)
        first = "The asset/variant pairs exceeding buy-and-hold's average yearly return after costs were " + descriptions + "."
    sentences = [first,
        f"Of the {len(assets)*len(VARIANTS)} asset/variant comparisons, {len(both_wins)} exceeded buy-and-hold on both average return and Sharpe, a measure of return relative to variability.",
        f"The turnover penalty reduced average daily trading on {penalty_less_trading} of {len(assets)} assets and improved average return over original PPO on {penalty_better_return} of {len(assets)}.",
        f"The long-only agent improved average return over original PPO on {long_better_return} of {len(assets)} assets, but buying a historically rising market does not itself show predictive timing.",
        f"The volatility-targeted holding rule had lower realized return variability than buy-and-hold on {vol_lower_risk} of {len(assets)} assets, although its target is not a guarantee of future risk.",
        f"The fixed-feature forecast improved average return over the original forecasting rule on {forecast_better_return} of {len(assets)} assets, which evaluates the combined fixed model and position rule rather than proving the inputs caused an improvement.",
        "The five seeds show variation from random starting points, while identical rule-based decisions have zero seed spread and neither measures uncertainty across future markets.",
        "Every variant was kept with its declared settings, but these reused test dates make this exploratory evidence that still needs new untouched history before claiming a lasting improvement."]
    paragraph = "\n".join(sentences)
    definitions = """A **seed** is a repeatable random starting point. **Net return** is growth after
fees and slippage (worse execution prices). **Annualized** means expressed on a
one-year compounded scale. **Volatility** is return variability. **Sharpe** divides
average return by variability and scales to a year. **Drawdown** is a fall from a
previous wealth peak. **Turnover** is the change in target exposure each day; 1.0
means an account's full target exposure changed. **Hit rate** is the fraction of
exposed periods with positive net return. Values following ± are sample standard
deviations: spread across seeds, not confidence about future markets.
"""
    body = "# Pre-declared improvement round: all outcomes\n\n" + definitions + "\n## Eight-sentence interpretation\n\n" + paragraph
    body += "\n\n## All strategies, after costs\n\n" + comparison_table(summary)
    body += f"\n\nThe plan was frozen at commit `{metadata['plan_commit']}` and implementation at `{metadata['code_commit']}` before fitting. Original results are reused from `results/final/`; all new paths and failures stay in `{output.as_posix()}/`. This round fitted {metadata['ppo_fits_total']} new PPO seed/window policies and evaluated all four variants on five seeds and 17 windows per asset. No settings were chosen from these outcomes.\n"
    body += "\n## Curves and exposure\n\nCurves display the mean of individual account paths; shading shows the full seed range. They are descriptive averages, not an implemented combined strategy. Drawdowns are calculated separately for each seed before display. `exposure_by_seed.csv` retains before-cost returns, trading activity, exposure sizes, and long/short/flat fractions.\n"
    body += "\n## What remains uncertain\n\nThe round follows inspection of earlier test results. Pre-declaration prevents additional searches within the round but does not undo that selection. The volatility target only roughly matches risk; long-only gains can reflect market direction. The forecast changed its loss, tree settings, and position scaling together. Vendor adjustments, survivor selection, shared calendars, price-drift rebalancing, funding, short borrowing, and simplified fills remain limitations. No future-profit or formal significance claim is made.\n"
    (output / "comparison.md").write_text(body + "\n")
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "IMPROVEMENT_RESULTS.md").write_text(body + "\n")
    summary_doc = docs / "RESULTS_SUMMARY.md"
    marker, end = "<!-- IMPROVEMENT_ROUND_001 -->", "<!-- END_IMPROVEMENT_ROUND_001 -->"
    block = marker + "\n\n" + body.replace("# Pre-declared", "## Pre-declared", 1) + "\n" + end
    old = summary_doc.read_text()
    if marker in old:
        before, rest = old.split(marker, 1)
        _, after = rest.split(end, 1)
        summary_doc.write_text(before + block + after)
    else:
        summary_doc.write_text(old + "\n\n" + block + "\n")
    findings = {"return_wins_vs_buy_hold": return_wins, "both_return_and_sharpe_wins": both_wins,
                "penalty_reduces_trading_assets": penalty_less_trading,
                "penalty_improves_original_ppo_return_assets": penalty_better_return,
                "long_only_improves_original_ppo_return_assets": long_better_return,
                "vol_target_reduces_variability_assets": vol_lower_risk,
                "forecast_improves_original_forecast_return_assets": forecast_better_return,
                "assets": len(assets), "strategies": len(summary.strategy.unique()),
                "seed_metric_rows": len(raw), "original_ppo_fits": 255,
                "new_ppo_fits": metadata["ppo_fits_total"], "sentences": sentences}
    (output / "findings.json").write_text(json.dumps(findings, indent=2))
    make_plots(output, raw)
    print(paragraph)
    return findings


if __name__ == "__main__":
    import yaml
    config = yaml.safe_load(Path("config.yaml").read_text())
    write_report(Path(config["improvement_round"]["output"]))
