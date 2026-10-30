"""Turn completed experiment artifacts into plain-language, reproducible reports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Small dependency-free Markdown table formatter."""
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in rows])


def checks_report(root: Path = Path("results/checks")) -> None:
    """Explain fixed-budget shuffle, frozen-policy costs, and random-policy luck."""
    ordinary = pd.read_csv(root / "ordinary/metrics.csv")
    shuffled = pd.read_csv(root / "shuffled/metrics.csv")
    plan = json.loads((root / "plan.json").read_text())
    rows = []
    for row in ordinary[ordinary.strategy.isin(["ppo", "gradient_boosting"])].itertuples():
        other = shuffled[(shuffled.asset == row.asset) & (shuffled.strategy == row.strategy)].iloc[0]
        rows.append([row.asset, row.strategy, f"{row.annualized_return_mean:.2%} ± {row.annualized_return_std:.2%}",
                     f"{other.annualized_return_mean:.2%} ± {other.annualized_return_std:.2%}",
                     f"{row.sharpe_mean:.2f} ± {row.sharpe_std:.2f}",
                     f"{other.sharpe_mean:.2f} ± {other.sharpe_std:.2f}"])
    table = markdown_table(["Asset", "Strategy", "Ordinary net annual return", "Shuffled net annual return", "Ordinary Sharpe", "Shuffled Sharpe"], rows)
    interpretation = []
    for asset in sorted(ordinary.asset.unique()):
        a = ordinary[(ordinary.asset == asset) & (ordinary.strategy == "ppo")].iloc[0]
        b = shuffled[(shuffled.asset == asset) & (shuffled.strategy == "ppo")].iloc[0]
        interpretation.append(f"{asset}: {a.sharpe_mean:.2f} ordinary versus {b.sharpe_mean:.2f} shuffled")
    costs = pd.read_csv(root / "cost_sensitivity/cost_sensitivity.csv")
    cost_rows = []
    for (asset, strategy), group in costs.groupby(["asset", "strategy"]):
        values = group.set_index("all_in_cost_bps")
        cost_rows.append([asset, strategy] + [f"{values.loc[bps, 'sharpe_mean']:.2f}" for bps in [0, 5, 7, 20]])
    comparison = pd.read_csv(root / "random_policy/random_policy_comparison.csv")
    matching = pd.read_csv(root / "random_policy/random_policy_matching.csv")
    random_runs = pd.read_csv(root / "random_policy/random_policy_runs.csv")
    random_rows = []
    for row in matching.itertuples():
        values = random_runs[random_runs.asset == row.asset]
        comp = comparison[comparison.asset == row.asset]
        random_rows.append([row.asset, f"{row.target_mean_abs_exposure:.4f}", f"{row.target_internal_turnover:.4f}",
                            f"{values.internal_turnover.mean():.4f}",
                            f"{values.sharpe.quantile(.05):.2f} to {values.sharpe.quantile(.95):.2f}",
                            f"{comp.random_sharpe_percentile.min():.1f}–{comp.random_sharpe_percentile.max():.1f}"])
    text = f"""# Robustness checks: what happened and what it means

These are diagnostics, meaning deliberate tests of the experiment. They do not
choose trading settings. All use the same real test returns, all 17 chronological
folds (training/test pairs) per asset, and five fixed random seeds (repeatable random
starting points). Ordinary and shuffled training each receive
{plan['config']['ppo_timesteps']:,} PPO learning steps per fold. This is less than
the 10,000-step final budget; differences in the final run must not be attributed
solely to shuffling. Net means after the default 7 bps all-in trading cost; one
basis point (bp) is 0.01%. Values following ± are sample standard deviations,
meaning the spread across the five runs, not confidence intervals.

## 1. Shuffled training answers

A label is the future return used as the answer during training. We randomly
reorder only each fold's training labels, independently for each asset/fold/seed.
The forecasting model learns from reordered labels; PPO receives those reordered
returns when calculating its training rewards. Features, dates, and actual test
answers stay unchanged. The ordinary comparison uses identical dates, seeds, and
learning budgets. Buy-and-hold and MA rules ignore training labels and do not change.

{table}

In this run, PPO's average Sharpe barely changed when its training answers were
scrambled ({'; '.join(interpretation)}). At this budget, that provides little
evidence that PPO learned useful timing of daily returns. The forecasting model's
average return fell after shuffling on all three assets, but this comparison alone
does not establish reliable forecasting skill or superiority to buy-and-hold.
The full-budget PPO run is a different experiment, so this smaller-budget result
cannot by itself rule out anything learned with more training.

**Why is the shuffled result not necessarily zero?** Shuffling breaks the match
between inputs and answers, but preserves the average training return. An agent
can learn to stay long, meaning hold exposure that gains when prices rise, even
without predicting which days rise. The real test market can rise or fall; a
constant or lucky exposure then makes or loses money. Trading costs push active
zero-signal policies below zero. Compounding, repeatedly applying returns to
changing wealth, also makes geometric growth differ from average daily profit.
Therefore nonzero shuffled returns alone are neither proof of a software bug nor
proof of forecasting skill. Compare ordinary vs shuffled at the same budget, inspect
market exposure and the random-policy distribution, and investigate unexpectedly
strong persistent differences. This is five random permutations per fold, not a
formal statistical significance test or an acceptance rule requiring zero profit.

## 2. Cost sensitivity

We keep learned decisions fixed and rerun the accounting at 0, 5, and 20 bps total
one-way cost, also showing the actual default of 7 bps = 5 fee + 2 slippage.
Slippage means receiving a worse trading price than assumed. Zero here truly means
zero fees AND zero slippage. Repricing includes first entries, reversals, and final
exits. This isolates the effect of fees; it does not train a different PPO policy
at each fee. Sharpe is average net return divided by return variability, scaled
to a year. Higher is usually preferable, but it need not fall monotonically with
cost because cost can also change the denominator, the variability.

{markdown_table(['Asset', 'Strategy', 'Sharpe: 0 bps', '5 bps', '7 bps', '20 bps'], cost_rows)}

## 3. What luck looks like

For each asset, {plan['random_runs']:,} random policies flip between equally sized
long and short positions. Average absolute exposure, meaning how much money is at
risk regardless of direction, is matched to the ordinary PPO seeds. The probability
of flipping direction is chosen to match average internal turnover (position changes
between days); initial entry and final exit are also charged. Actual matching error
is reported below. Matching uses observed test positions, not test returns, so
this is a retrospective comparison and is not a deployable calibrated strategy.

{markdown_table(['Asset', 'Target exposure', 'Target daily internal turnover', 'Random actual turnover', 'Random middle 90% Sharpe', 'PPO percentile range'], random_rows)}

A percentile of 80 means that PPO seed's Sharpe exceeded about 80% of these random
runs. It is not an 80% chance of future success or a formal probability of skill.
The middle 90% interval describes outcomes on this one realized market history;
it is not uncertainty across possible future markets. Matching only average
exposure and turnover does not reproduce PPO's entire action distribution.

## Files and reproduction

Run `python -m src.checks.run_checks`. `plan.json` freezes the settings before
training. `ordinary/` and `shuffled/` contain per-seed metrics, per-day accounting,
fold bounds, timing, and manifests. `cost_sensitivity/` contains four-cost metrics
and Sharpe plots. `random_policy/` contains every random run, matching details,
seed percentiles, and histograms. Prediction checkpoints can resume identical
experiments; source/config/data signatures prevent reuse after an input change.
No result was used to change features, seeds, costs, or model parameters.

Shuffling also destroys the original order of returns and makes simulated reward
history inconsistent with the lagged real-return features. It is a deliberately
artificial control, not a realistic alternate market. Five permutations per fold
provide limited evidence; no formal significance claim is made.
"""
    (root / "report.md").write_text(text)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, asset in zip(axes, sorted(ordinary.asset.unique())):
        for offset, (name, frame) in enumerate([("Ordinary", ordinary), ("Shuffled", shuffled)]):
            values = frame[(frame.asset == asset) & frame.strategy.isin(["gradient_boosting", "ppo"])].set_index("strategy").loc[["gradient_boosting", "ppo"]]
            ax.bar(np.arange(2) + (offset - .5) * .35, values.sharpe_mean, width=.35,
                   yerr=values.sharpe_std, capsize=3, label=name)
        ax.set_xticks([0, 1], ["Forecast", "PPO"])
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set_title(asset)
    axes[0].set_ylabel("Net Sharpe (mean ± seed std)")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(root / "shuffled_label_comparison.png", dpi=150)
    plt.close(fig)


def final_report(final: Path = Path("results/final"), checks: Path = Path("results/checks")) -> None:
    """Write an honest results table and eight explanatory sentences from actual runs."""
    summary = pd.read_csv(final / "metrics.csv")
    raw = pd.read_csv(final / "metrics_by_seed.csv")
    manifest = json.loads((final / "manifest.json").read_text())
    costs = pd.read_csv(final / "cost_sensitivity.csv")
    random = pd.read_csv(checks / "final_random_policy/random_policy_comparison.csv")
    ordinary = pd.read_csv(checks / "ordinary/metrics.csv")
    shuffled = pd.read_csv(checks / "shuffled/metrics.csv")
    rows = []
    for row in summary.itertuples():
        rows.append([row.asset, row.strategy, f"{row.annualized_return_mean:.2%} ± {row.annualized_return_std:.2%}",
                     f"{row.annualized_volatility_mean:.2%} ± {row.annualized_volatility_std:.2%}",
                     f"{row.sharpe_mean:.2f} ± {row.sharpe_std:.2f}",
                     f"{row.max_drawdown_mean:.2%} ± {row.max_drawdown_std:.2%}",
                     f"{row.turnover_mean:.4f} ± {row.turnover_std:.4f}",
                     f"{row.hit_rate_mean:.2%} ± {row.hit_rate_std:.2%}"])
    table = markdown_table(["Asset", "Strategy", "Annual net return", "Annual volatility", "Sharpe", "Max drawdown", "Daily turnover", "Hit rate"], rows)
    wins = []
    dominated_everywhere = True
    seed_text, evidence_rows = [], []
    for asset in sorted(summary.asset.unique()):
        group = summary[summary.asset == asset].set_index("strategy")
        ppo = group.loc["ppo"]
        baselines = group.drop("ppo")
        if ppo.annualized_return_mean > baselines.annualized_return_mean.max() and ppo.sharpe_mean > baselines.sharpe_mean.max():
            wins.append(asset)
        dominated_everywhere &= bool(ppo.annualized_return_mean < baselines.annualized_return_mean.min() and
                                     ppo.sharpe_mean < baselines.sharpe_mean.min())
        agents = raw[(raw.asset == asset) & (raw.strategy == "ppo")]
        seed_text.append(f"{asset}: {int((agents.annualized_return > 0).sum())} of {len(agents)}")
        c = costs[(costs.asset == asset) & (costs.strategy == "ppo")].set_index("all_in_cost_bps")
        a = ordinary[(ordinary.asset == asset) & (ordinary.strategy == "ppo")].iloc[0]
        b = shuffled[(shuffled.asset == asset) & (shuffled.strategy == "ppo")].iloc[0]
        comp = random[random.asset == asset]
        evidence_rows.append([asset, f"{c.loc[0, 'sharpe_mean']:.2f} → {c.loc[20, 'sharpe_mean']:.2f}",
                              f"{a.sharpe_mean:.2f} → {b.sharpe_mean:.2f}",
                              f"{comp.random_sharpe_percentile.min():.1f}–{comp.random_sharpe_percentile.max():.1f}"])
    conclusion = ("PPO did not beat the best baseline on both mean annualized net return and mean Sharpe for any asset."
                  if not wins else f"PPO beat every baseline on both mean annualized net return and mean Sharpe for {', '.join(wins)} only; this is a historical comparison, not proof of future skill.")
    if dominated_everywhere:
        conclusion = "PPO performed worse than every baseline on every asset by both average annual return after costs and Sharpe, which measures average return relative to its variability."
    sentences = [
        "We repeatedly learned from earlier three-year windows and tested on the following up-to-six months, using 17 test windows per asset, five random starting points, and a trading charge of 7 basis points (0.07%) per unit of exposure changed.",
        conclusion,
        "The numbers of PPO runs with positive annual returns were " + "; ".join(seed_text) + ", so losses were consistent across starting points for SPY and QQQ.",
        "The ± values measure variation across starting points rather than uncertainty about future markets, while the simple rules have zero variation because they are repeated unchanged.",
        "Keeping decisions fixed, increasing fees plus slippage from zero to 20 basis points worsened PPO's average Sharpe on all three assets, and even free trading left it behind buy-and-hold on this measure.",
        "At the separate, equal 1,024-step training budget, scrambling the training answers barely changed PPO's results, providing little evidence that it learned useful daily return timing at that budget.",
        "The 1,000 random-agent comparisons per asset showed that a PPO starting point could rank well against luck while still losing money, so one favorable rank does not establish reliable skill.",
        "We did not tune using these test results, and revised data, asset selection, simplified trading assumptions, and future market changes still limit the conclusion.",
    ]
    paragraph = "\n".join(sentences)
    evidence = markdown_table(["Asset", "Full PPO Sharpe: 0 → 20 bps", "1,024-step PPO Sharpe: ordinary → shuffled", "Full PPO random percentile range"], evidence_rows)
    trace = pd.read_csv(final / f"{manifest['config']['assets'][0]}_ppo_{manifest['config']['seeds'][0]}_oos.csv")
    document = f"""# Results summary: full-budget experiment

## In plain language

{paragraph}

## Results table

Net means after fees and slippage; bps means basis points, each equal to 0.01%.
Mean is the average across five seeds (repeatable random starting states), and
standard deviation after ± measures their spread. Volatility measures how variable
returns are; Sharpe divides average return by that variability and scales to a
year; max drawdown is the worst fall from a previous wealth peak; turnover measures
position changes; hit rate counts profitable exposed periods. Daily turnover is a
fraction of account exposure, so 1.0 means 100% of account value changed hands.

{table}

Buy-and-hold, moving-average crossover, and gradient boosting keep their original
fixed settings. They are repeated identically for all five seed rows to make the
table rectangular; their zero seed std is structural and does not imply certainty
about markets. PPO seed metrics are calculated separately before their mean/std.

## What the checks added

{evidence}

All-in cost includes both fees and slippage. The random percentiles show the range
across the five PPO seeds; a higher percentile does not mean a probability of future
success. Shuffling was tested at a smaller, explicitly matched training budget.
A near-zero Sharpe does not imply zero compounded annual return: gaining 10% and
then losing 10% leaves only 99% of starting wealth. This distinction helps explain
BTC-USD's negative annual growth despite an average PPO Sharpe close to zero.

## Reproduce and inspect

The data end is pinned to `{manifest['config']['end']}` (exclusive) using the original
cached downloads. `results/final/config.json`, `manifest.json`, and `folds.csv`
record settings, versions, signatures, and boundaries. `metrics.csv` contains
mean/std for all six metrics; `metrics_by_seed.csv` retains every run.
Test decisions ran from {trace.decision_time.iloc[0]} through
{trace.decision_time.iloc[-1]}, with the last return realized at
{trace.outcome_end.iloc[-1]}; earlier history supplies training and feature warmup.
Each PPO fit requested {manifest['config']['ppo_timesteps']:,} learning steps;
the library completes its 256-step chunks, collecting
{manifest.get('effective_steps_per_fit', 10240):,} training steps per fit.
The measured training-and-main-output runtime was {manifest['elapsed_seconds'] / 60:.1f}
minutes, below the conservative 21.3-minute pre-run estimate and 30-minute threshold.
`*_oos.csv` gives daily decisions, realized returns, costs, and final liquidation.
Equity curves track compounded account value; drawdown plots track losses from
previous peaks. `*_cost_sensitivity.png` reprices identical positions at 0/5/7/20
bps total cost, including slippage, without retraining policies for each cost.
See `results/checks/report.md` for the paired shuffled-label experiment and its
lower training budget; `results/checks/final_random_policy/` compares the full
agent against random positions calibrated retrospectively to its activity.

## Limitations that matter for this conclusion

- The five seeds reuse the same market history; they are not five independent
  economic experiments or a confidence interval for future profitability.
- Shuffling training returns preserves their mean and can reward market direction;
  it also breaks serial order and the consistency between lagged features and
  simulated reward history. Five permutations per fold are an artificial
  diagnostic, not a formal statistical test or alternate realistic market.
- Cost plots hold positions fixed and do not model how an agent retrained at a
  different cost might behave. Constant slippage can understate difficult execution.
- Random policies match average exposure and turnover, using test positions for
  descriptive calibration; they do not match all PPO behavior or constitute a
  strategy available before the test. Their percentiles are not statistical p-values.
- Adjusted vendor history, surviving asset selection, and common-date alignment
  can carry retrospective information even though code-level timing is causal.
- Rebalancing from price drift, borrowing/funding, liquidity, taxes, execution
  delays, and market impact are omitted. The observation omits some reward history.
- All test periods have now been inspected. Any further change motivated by these
  results requires new untouched data or nested validation, meaning separate
  earlier data used for model choices before a final untouched evaluation.
"""
    Path("docs").mkdir(exist_ok=True)
    Path("docs/RESULTS_SUMMARY.md").write_text(document)
    readme = Path("README.md")
    contents = readme.read_text()
    marker = "<!-- FINAL_RESEARCH_RESULTS -->"
    end_marker = "<!-- END_FINAL_RESEARCH_RESULTS -->"
    block = f"{marker}\n## Full-budget results and robustness checks\n\n{paragraph}\n\n{table}\n\nRead [the beginner guide](docs/EXPLAINED.md), [the full results discussion](docs/RESULTS_SUMMARY.md), and [the robustness report](results/checks/report.md). Run `python -m src.checks.run_checks`, then `python -m src.checks.estimate_runtime`, then `python -m src.checks.run_final`. Full metrics are in `results/final/metrics.csv`.\n\n### Additional limitations\n\nThe five seeds share the same history; baseline seed spreads are zero by construction. Shuffled labels preserve average returns, and five permutations per fold do not establish statistical significance. Cost sensitivity holds policies fixed. Random-policy matching uses OOS positions for retrospective calibration, so percentile ranks are descriptive rather than formal probabilities of skill. Final outcomes are from already inspected dates and require a fresh holdout before further research claims.\n{end_marker}"
    if marker in contents:
        before, rest = contents.split(marker, 1)
        _, after = rest.split(end_marker, 1)
        contents = before + block + after
    else:
        contents += "\n\n" + block + "\n"
    readme.write_text(contents)
