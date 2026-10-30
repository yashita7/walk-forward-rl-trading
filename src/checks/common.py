"""Shared experiment loading, summaries, and auditable fold evaluation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from ..backtest import account, metrics, walk_forward
from ..baselines import baseline_positions
from ..data import align_bars, download_bars
from ..features import decision_frame
from ..run_all import plot_results
from ..train_rl import train_predict

METRICS = ["annualized_return", "annualized_volatility", "sharpe", "max_drawdown", "turnover", "hit_rate"]


def load_config(path: str = "config.yaml") -> dict:
    """Pin the cached end date for repeatability; do not silently extend the sample."""
    with open(path) as stream:
        config = yaml.safe_load(stream)
    if config.get("end") is None:
        manifest = Path("results/market_smoke/manifest.json")
        if not manifest.exists():
            raise ValueError("Set an explicit end date or first create the original market-smoke manifest")
        config["end"] = json.loads(manifest.read_text())["config"]["end"]
    config["seeds"] = [7, 21, 42, 84, 123]
    return config


def load_data(config: dict) -> dict[str, pd.DataFrame]:
    """Use the same cached bars and common session calendar as the normal runner."""
    return align_bars({asset: download_bars(asset, config["start"], config["end"], config["data_dir"])
                       for asset in config["assets"]})


def shuffled_training_returns(y: pd.Series, train: np.ndarray, seed: int) -> pd.Series:
    """Permute training answers only; preserve test answers and training distribution."""
    result = y.copy()
    result.iloc[train] = np.random.default_rng(seed).permutation(y.iloc[train].to_numpy())
    return result


def summarize(raw: pd.DataFrame, groups: list[str] | None = None) -> pd.DataFrame:
    """Flatten mean/std columns; repeated deterministic baselines have structural std 0."""
    groups = groups or ["asset", "strategy"]
    summary = raw.groupby(groups)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    counts = raw.groupby(groups).seed.nunique().rename("seed_count")
    return summary.join(counts).reset_index()


def evaluate(config: dict, output: Path, *, shuffle: bool = False) -> pd.DataFrame:
    """Run every fold/seed; checkpoint predictions to resume identical experiments.

    No test outcome enters model fitting. Shuffled labels are independently
    permuted per asset/fold/seed. Models always earn actual unshuffled OOS returns.
    """
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    data = load_data(config)
    source = b"".join(p.read_bytes() for p in sorted(Path("src").rglob("*.py")))
    signature = hashlib.sha256(source + json.dumps(config, sort_keys=True).encode() + str(shuffle).encode())
    for bars in data.values():
        signature.update(pd.util.hash_pandas_object(bars, index=True).values.tobytes())
    digest = signature.hexdigest()
    raw_rows, folds_audit, timings = [], [], []
    all_started = perf_counter()
    for asset_number, (asset, bars) in enumerate(data.items()):
        x, y, ends = decision_frame(bars)
        folds = list(walk_forward(x.index, ends, config["train_years"], config["test_months"],
                                  config["observation_window"]))
        if not folds:
            raise ValueError(f"{asset}: no folds")
        paths: dict[tuple[str, int], list[pd.DataFrame]] = {}
        previous: dict[tuple[str, int], float] = {}
        for fold in folds:
            folds_audit.append({"asset": asset, "fold": fold.number,
                                "train_first": x.index[fold.train[0]], "train_last": x.index[fold.train[-1]],
                                "last_training_outcome": ends.iloc[fold.train].max(),
                                "test_first": x.index[fold.test[0]], "test_last": x.index[fold.test[-1]],
                                "test_boundary": fold.test_start, "train_rows": len(fold.train), "test_rows": len(fold.test)})
            print(f"{output.name}: {asset} fold {fold.number + 1}/{len(folds)}", flush=True)
            fixed = baseline_positions(x.iloc[fold.train], y.iloc[fold.train], x.iloc[fold.test], config["forecast_seed"])
            for seed in config["seeds"]:
                key = ("ppo", seed)
                prior = previous.get(key, 0.0)
                permutation_seed = int(np.random.SeedSequence([seed, asset_number, fold.number, 891]).generate_state(1)[0])
                training_y = shuffled_training_returns(y, fold.train, permutation_seed) if shuffle else y
                name = checkpoint / f"{asset}_fold{fold.number}_seed{seed}_{digest[:16]}.npz"
                started = perf_counter()
                cached = name.exists()
                if cached:
                    with np.load(name) as stored:
                        if float(stored["previous"]) != prior:
                            raise ValueError("Checkpoint previous position mismatch")
                        ppo = stored["ppo"]
                        forecast = stored["forecast"]
                else:
                    ppo = train_predict(x, training_y, fold, config, seed, prior)
                    forecast = baseline_positions(x.iloc[fold.train], training_y.iloc[fold.train],
                                                  x.iloc[fold.test], config["forecast_seed"])["gradient_boosting"] if shuffle else fixed["gradient_boosting"]
                    np.savez_compressed(name, ppo=ppo, forecast=forecast, previous=prior)
                timings.append({"asset": asset, "fold": fold.number, "seed": seed,
                                "cached": cached, "seconds": perf_counter() - started})
                positions = {**fixed, "gradient_boosting": forecast, "ppo": ppo}
                for strategy, values in positions.items():
                    strategy_key = (strategy, seed)
                    frame = account(values, y.iloc[fold.test], ends.iloc[fold.test],
                                    (config["transaction_cost_bps"] + config["slippage_bps"]) / 10000,
                                    previous.get(strategy_key, 0), liquidate=fold.number == folds[-1].number)
                    frame["fold"] = fold.number
                    paths.setdefault(strategy_key, []).append(frame)
                    previous[strategy_key] = float(values[-1])
        complete = {key: pd.concat(parts) for key, parts in paths.items()}
        for (strategy, seed), frame in complete.items():
            if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
                raise AssertionError("OOS dates must be ordered and unique")
            frame.index.name = "decision_time"
            frame.to_csv(output / f"{asset}_{strategy}_{seed}_oos.csv")
            raw_rows.append({"asset": asset, "strategy": strategy, "seed": seed,
                             "observations": len(frame), **metrics(frame, config["annualization"])})
        # Deterministic baselines overlap exactly; draw them once, PPO for every seed.
        display = {key: frame for key, frame in complete.items() if key[0] == "ppo" or key[1] == config["seeds"][0]}
        plot_results(display, asset, output)
        pd.DataFrame(raw_rows).to_csv(output / "metrics_by_seed.csv", index=False)
    raw = pd.DataFrame(raw_rows)
    summarize(raw).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(folds_audit).to_csv(output / "folds.csv", index=False)
    pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)
    packages = ["numpy", "pandas", "scikit-learn", "stable-baselines3", "torch", "gymnasium", "yfinance"]
    manifest = {"config": config, "shuffle_training_returns": shuffle, "signature": digest,
                "elapsed_seconds": perf_counter() - all_started,
                "effective_steps_per_fit": int(np.ceil(config["ppo_timesteps"] / config["ppo_n_steps"]) * config["ppo_n_steps"]) if "ppo_timesteps" in config else None,
                "versions": {p: importlib.metadata.version(p) for p in packages},
                "baseline_seed_note": "Deterministic baselines repeated identically across seeds; zero std is structural."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return raw
