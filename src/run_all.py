"""CLI: python -m src.run_all [--smoke] [--synthetic]."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .backtest import account, equity_and_drawdown, metrics, walk_forward
from .baselines import baseline_positions
from .data import align_bars, download_bars, synthetic_bars
from .features import decision_frame
from .train_rl import train_predict


def plot_results(paths: dict[tuple[str, int], pd.DataFrame], asset: str, output: Path) -> None:
    """Draw every RL seed, preserving seed variation in the plots."""
    for kind in ("equity", "drawdown", "positions"):
        fig, ax = plt.subplots(figsize=(11, 5))
        for (strategy, seed), frame in paths.items():
            if kind == "positions" and strategy != "ppo":
                continue
            equity, drawdown = equity_and_drawdown(frame.net_return)
            values = {"equity": equity, "drawdown": drawdown, "positions": frame.position}[kind]
            dates = frame.index if kind == "positions" else pd.DatetimeIndex(frame.outcome_end)
            ax.plot(dates, values, label=f"{strategy}" + (f" seed={seed}" if strategy == "ppo" else ""),
                    linewidth=1, alpha=0.85)
        ax.set(title=f"{asset}: out-of-sample {kind}", ylabel=kind, xlabel="Date")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{asset}_{kind}.png", dpi=150)
        plt.close(fig)


def run(config: dict, *, synthetic: bool = False, smoke: bool = False) -> pd.DataFrame:
    """Evaluate per asset (no pooled portfolio), save audit traces and seed summaries."""
    config = dict(config)
    if smoke:
        config.update(ppo_timesteps=256, ppo_n_steps=64, ppo_batch_size=32)
    end = config.get("end") or datetime.now(timezone.utc).date().isoformat()
    if pd.Timestamp(end) > pd.Timestamp(datetime.now(timezone.utc).date()):
        raise ValueError("End must not include future/incomplete daily bars")
    config["end"] = end
    output = Path(config["results_dir"]) / ("synthetic_smoke" if synthetic else "market_smoke" if smoke else "market")
    output.mkdir(parents=True, exist_ok=True)
    data = {asset: synthetic_bars(100+i) if synthetic else download_bars(
        asset, config["start"], end, config["data_dir"])
        for i, asset in enumerate(config["assets"])}
    data = align_bars(data)
    rows, audits = [], []
    rate = (config["transaction_cost_bps"] + config["slippage_bps"]) / 10000
    if rate < 0 or config["transaction_cost_bps"] < 0 or config["slippage_bps"] < 0:
        raise ValueError("Costs and slippage must be nonnegative")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty, unique RL seeds")
    for asset, bars in data.items():
        x, y, ends = decision_frame(bars)
        folds = list(walk_forward(x.index, ends, config["train_years"],
                                  config["test_months"], config["observation_window"]))
        if smoke:
            folds = folds[:2]
        if not folds:
            raise ValueError(f"{asset}: insufficient history for a walk-forward fold")
        paths: dict[tuple[str, int], list[pd.DataFrame]] = {}
        previous: dict[tuple[str, int], float] = {}
        for fold in folds:
            print(f"{asset} fold {fold.number}: train {len(fold.train)}, test {len(fold.test)}", flush=True)
            audits.append({"asset": asset, "fold": fold.number,
                           "train_first": x.index[fold.train[0]], "train_last": x.index[fold.train[-1]],
                           "last_training_outcome": ends.iloc[fold.train].max(),
                           "test_first": x.index[fold.test[0]], "test_last": x.index[fold.test[-1]],
                           "test_boundary": fold.test_start})
            positions = {(name, -1): values for name, values in baseline_positions(
                x.iloc[fold.train], y.iloc[fold.train], x.iloc[fold.test], config["forecast_seed"]).items()}
            for seed in seeds:
                key = ("ppo", seed)
                positions[key] = train_predict(x, y, fold, config, seed, previous.get(key, 0.0))
            for key, values in positions.items():
                frame = account(values, y.iloc[fold.test], ends.iloc[fold.test], rate,
                                previous.get(key, 0.0), liquidate=fold.number == folds[-1].number)
                frame["fold"] = fold.number
                paths.setdefault(key, []).append(frame)
                previous[key] = float(values[-1])
        completed = {key: pd.concat(parts).sort_index() for key, parts in paths.items()}
        for (strategy, seed), frame in completed.items():
            if frame.index.has_duplicates:
                raise AssertionError("Overlapping OOS periods")
            frame.index.name = "decision_time"
            frame.to_csv(output / f"{asset}_{strategy}_{seed}_oos.csv")
            rows.append({"asset": asset, "strategy": strategy, "seed": seed,
                         "observations": len(frame), **metrics(frame, config["annualization"])})
        plot_results(completed, asset, output)
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)
    metric_names = ["annualized_return", "annualized_volatility", "sharpe", "max_drawdown", "turnover", "hit_rate"]
    summary = table.groupby(["asset", "strategy"])[metric_names].agg(["mean", "std"])
    summary.to_csv(output / "metrics_summary.csv")
    pd.DataFrame(audits).to_csv(output / "folds.csv", index=False)
    packages = ["numpy", "pandas", "scikit-learn", "gymnasium", "stable-baselines3", "torch", "yfinance"]
    manifest = {"config": config, "synthetic": synthetic, "smoke": smoke,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "versions": {p: importlib.metadata.version(p) for p in packages},
                "data_ranges": {a: {"first": str(b.index.min()), "last": str(b.index.max()), "rows": len(b)} for a,b in data.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(table.to_string(index=False))
    print(f"Saved results to {output.resolve()}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Two folds, 256 PPO steps, all assets/seeds")
    parser.add_argument("--synthetic", action="store_true", help="Explicit offline fixture; not market performance")
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    run(config, synthetic=args.synthetic, smoke=args.smoke)


if __name__ == "__main__":
    main()
