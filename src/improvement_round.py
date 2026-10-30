"""Execute a committed fixed round once, preserving predictions, errors, and logs.

Run: python -m src.improvement_round
Only an interrupted, identical run may use --resume. No automatic parameter search.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from time import perf_counter
import traceback

import numpy as np
import pandas as pd
import yaml

from .backtest import account, metrics, walk_forward
from .checks.common import load_data, summarize
from .diagnostics import position_statistics
from .round_variants import VARIANTS, forecast_positions, ppo_configuration, volatility_target_positions
from .features import decision_frame
from .train_rl import train_predict


def git_read(*args: str) -> str:
    """Read local version metadata; never change Git state from the experiment."""
    return subprocess.check_output(["git", *args], text=True).strip()


def committed_plan() -> dict[str, str]:
    """Reject an uncommitted plan or implementation before fitting a model."""
    protected = ["config.yaml", "docs/EXPERIMENT_PLAN.md", "src", "tests", "requirements.txt", "requirements-lock.txt", "pyproject.toml"]
    dirty = git_read("status", "--porcelain", "--", *protected)
    if dirty:
        raise ValueError("Commit the frozen plan/config/code/tests before training; protected files changed")
    commit = git_read("log", "-1", "--format=%H", "--", "config.yaml", "docs/EXPERIMENT_PLAN.md")
    if not commit:
        raise ValueError("No committed plan")
    for name in ["config.yaml", "docs/EXPERIMENT_PLAN.md"]:
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"])
        if content != Path(name).read_bytes():
            raise ValueError("Current plan differs from its declared commit")
    return {"plan_commit": commit, "code_commit": git_read("rev-parse", "HEAD")}


def audit_original(frame: pd.DataFrame, expected: pd.DatetimeIndex, actual_returns: pd.Series) -> None:
    """Original and new strategies must use identical ordered test decisions/prices."""
    if not frame.index.equals(expected):
        raise ValueError("Original/new OOS dates differ; refuse comparison")
    np.testing.assert_allclose(frame.asset_return, actual_returns.loc[expected], atol=1e-12, rtol=1e-12)


def event(output: Path, kind: str, **fields: object) -> None:
    """Append every fit/status/error event; never erase a prior attempt."""
    with (output / "ledger.jsonl").open("a") as stream:
        stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}) + "\n")


def checkpoint_positions(path: Path, previous: float) -> np.ndarray:
    """Load a prediction checkpoint only when the carried position matches."""
    with np.load(path) as stored:
        if float(stored["previous"]) != previous:
            raise ValueError("Checkpoint position mismatch")
        return stored["positions"]


def run(config: dict, output: Path, *, resume: bool = False) -> None:
    """Record the immutable run, fit the four hypotheses, and retain all seed paths."""
    if output.exists() and not resume:
        raise FileExistsError("This experiment already exists; refuse overwrite. Only --resume an interrupted identical run.")
    if resume and (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text()).get("status") == "complete":
            raise FileExistsError("Completed rounds are immutable; refuse another fit")
    output.mkdir(parents=True, exist_ok=resume)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    started = perf_counter()
    event(output, "resume_requested" if resume else "round_created")
    try:
        version = committed_plan()
        settings = config["improvement_round"]
        if tuple(settings["variants"]) != VARIANTS or len(config["seeds"]) < 5:
            raise ValueError("Exactly the four predeclared variants and at least five seeds are required")
        data = load_data(config)
        source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path("src").rglob("*.py")))).hexdigest()
        config_hash = hashlib.sha256(Path("config.yaml").read_bytes()).hexdigest()
        data_hash = hashlib.sha256()
        for asset, bars in data.items():
            data_hash.update(asset.encode())
            data_hash.update(pd.util.hash_pandas_object(bars, index=True).values.tobytes())
        package_names = ["numpy", "pandas", "scikit-learn", "torch", "gymnasium", "stable-baselines3", "yfinance"]
        immutable = {"source_sha256": source_hash, "config_sha256": config_hash,
                     "data_sha256": data_hash.hexdigest(), "config": config,
                     "versions": {name: importlib.metadata.version(name) for name in package_names}}
        manifest_path = output / "manifest.json"
        if resume:
            old = json.loads(manifest_path.read_text())
            if old.get("status") == "complete":
                raise ValueError("A completed round cannot be fitted again")
            if any(old[key] != value for key, value in immutable.items()):
                raise ValueError("Resume requires identical source, configuration, data, and library versions")
            version = {key: old[key] for key in version}
        manifest = {**immutable, **version, "status": "running", "exploratory_reused_test_dates": True}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        (output / "config_snapshot.yaml").write_bytes(Path("config.yaml").read_bytes())
        (output / "plan_snapshot.md").write_bytes(Path("docs/EXPERIMENT_PLAN.md").read_bytes())
        original = Path(settings["original_results"])
        original_manifest = json.loads((original / "manifest.json").read_text())
        for key in ["assets", "start", "end", "transaction_cost_bps", "slippage_bps", "train_years", "test_months", "observation_window", "risk_window", "volatility_penalty", "ppo_timesteps", "ppo_n_steps", "ppo_batch_size", "annualization", "seeds"]:
            if original_manifest["config"][key] != config[key]:
                raise ValueError(f"Original evaluation setting differs: {key}")
        original_raw = pd.read_csv(original / "metrics_by_seed.csv")
        raw_rows, exposure_rows, folds_rows, timing_rows = [], [], [], []
        rate = (config["transaction_cost_bps"] + config["slippage_bps"]) / 10000
        for asset, bars in data.items():
            x, y, ends = decision_frame(bars)
            folds = list(walk_forward(x.index, ends, config["train_years"], config["test_months"], config["observation_window"]))
            expected = x.index[np.concatenate([f.test for f in folds])]
            complete: dict[tuple[str, int], pd.DataFrame] = {}
            for row in original_raw[original_raw.asset == asset].itertuples():
                frame = pd.read_csv(original / f"{asset}_{row.strategy}_{row.seed}_oos.csv",
                                    parse_dates=["decision_time", "outcome_end"]).set_index("decision_time")
                audit_original(frame, expected, y)
                complete[(row.strategy, row.seed)] = frame
            required = {(strategy, seed) for strategy in ("buy_hold", "ma_50_200", "gradient_boosting", "ppo")
                        for seed in config["seeds"]}
            if set(complete) != required:
                raise ValueError("Original strategy/seed paths are missing or unexpected")
            vol_positions = volatility_target_positions(bars, config).reindex(x.index)
            previous: dict[tuple[str, int], float] = {}
            parts: dict[tuple[str, int], list[pd.DataFrame]] = {}
            for fold in folds:
                print(f"{settings['id']}: {asset} window {fold.number+1}/{len(folds)}", flush=True)
                folds_rows.append({"asset": asset, "fold": fold.number,
                                   "train_first": x.index[fold.train[0]], "train_last": x.index[fold.train[-1]],
                                   "last_training_outcome": ends.iloc[fold.train].max(),
                                   "test_boundary": fold.test_start, "test_first": x.index[fold.test[0]],
                                   "test_last": x.index[fold.test[-1]], "train_rows": len(fold.train), "test_rows": len(fold.test)})
                for seed in config["seeds"]:
                    for variant in VARIANTS:
                        key = (variant, seed)
                        prior = previous.get(key, 0.)
                        checkpoint = output / "checkpoints" / f"{asset}_{variant}_{seed}_fold{fold.number}.npz"
                        begin = perf_counter()
                        cached = checkpoint.exists()
                        fields = {"asset": asset, "variant": variant, "seed": seed, "fold": fold.number}
                        if cached:
                            positions = checkpoint_positions(checkpoint, prior)
                            event(output, "checkpoint_loaded", **fields)
                        else:
                            event(output, "fit_started" if variant != "vol_target_buy_hold" else "rule_started", **fields)
                            if variant.startswith("ppo_"):
                                positions = train_predict(x, y, fold, ppo_configuration(config, variant), seed, prior)
                            elif variant == "gb_fixed_features":
                                positions = forecast_positions(x.iloc[fold.train], y.iloc[fold.train], x.iloc[fold.test], config, seed)
                            else:
                                positions = vol_positions.iloc[fold.test].to_numpy()
                            np.savez_compressed(checkpoint, positions=positions, previous=prior)
                            event(output, "prediction_saved", **fields)
                        frame = account(positions, y.iloc[fold.test], ends.iloc[fold.test], rate, prior,
                                        liquidate=fold.number == folds[-1].number)
                        if variant in ("ppo_long_only", "vol_target_buy_hold") and (frame.position < 0).any():
                            raise AssertionError("A long-only strategy went short")
                        frame["fold"] = fold.number
                        parts.setdefault(key, []).append(frame)
                        previous[key] = float(positions[-1])
                        timing_rows.append({**fields, "cached": cached, "seconds": perf_counter()-begin})
            complete.update({key: pd.concat(value) for key, value in parts.items()})
            for (strategy, seed), frame in complete.items():
                audit_original(frame, expected, y)
                frame.index.name = "decision_time"
                frame.to_csv(output / f"{asset}_{strategy}_{seed}_oos.csv")
                raw_rows.append({"asset": asset, "strategy": strategy, "seed": seed, "observations": len(frame),
                                 **metrics(frame, config["annualization"])})
                exposure_rows.append({"asset": asset, "strategy": strategy, "seed": seed,
                                      **position_statistics(frame, config["annualization"])})
            pd.DataFrame(raw_rows).to_csv(output / "metrics_by_seed.csv", index=False)
            pd.DataFrame(exposure_rows).to_csv(output / "exposure_by_seed.csv", index=False)
        raw = pd.DataFrame(raw_rows)
        summarize(raw).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(folds_rows).to_csv(output / "folds.csv", index=False)
        pd.DataFrame(timing_rows).to_csv(output / "timings.csv", index=False)
        manifest.update(status="complete", elapsed_seconds=perf_counter()-started,
                        effective_ppo_steps_per_fit=int(np.ceil(config["ppo_timesteps"]/config["ppo_n_steps"])*config["ppo_n_steps"]),
                        ppo_fits_total=sum(1 for t in timing_rows if str(t["variant"]).startswith("ppo_")),
                        new_ppo_fits_this_invocation=sum(1 for t in timing_rows if str(t["variant"]).startswith("ppo_") and not t["cached"]))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        event(output, "round_completed", elapsed_seconds=manifest["elapsed_seconds"])
        print(f"Saved all original and new outcomes under {output}", flush=True)
    except Exception as exc:
        message = str(exc).replace(str(Path.cwd()), "<repo>")
        event(output, "round_failed", error_type=type(exc).__name__, message=message)
        number = len(list((output / "logs").glob("failure_*.txt"))) + 1
        (output / "logs" / f"failure_{number:03}.txt").write_text(traceback.format_exc())
        (output / f"failure_{number:03}.json").write_text(json.dumps({"error_type": type(exc).__name__, "message": message}, indent=2))
        if (output / "manifest.json").exists():
            failed = json.loads((output / "manifest.json").read_text())
            failed["status"] = "failed"
            (output / "manifest.json").write_text(json.dumps(failed, indent=2))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path("config.yaml").read_text())
    run(config, Path(config["improvement_round"]["output"]), resume=args.resume)


if __name__ == "__main__":
    main()
