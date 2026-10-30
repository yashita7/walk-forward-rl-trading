"""Small public result archives: positions/realized returns, never OHLCV price caches.

Export once: python -m src.result_archive --source results/final --export
Restore excluded daily CSVs: python -m src.result_archive --source results/final
Neither operation trains, changes parameters, or overwrites an existing history.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .backtest import account, metrics


def safe_label(value: str) -> str:
    """Only result labels, never path components, may come from archive metadata."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid archive result label")
    return value


def export(source: Path) -> list[Path]:
    """Archive each asset separately; keep baseline results usable in a fresh clone."""
    index = source / "history_archives.json"
    if index.exists():
        raise FileExistsError("Result archive already exists; refuse overwrite")
    raw = pd.read_csv(source / "metrics_by_seed.csv")
    config = json.loads((source / "manifest.json").read_text())["config"]
    archives = []
    checksums = {}
    for asset, group in raw.groupby("asset"):
        safe_label(asset)
        arrays, records = {}, []
        for number, row in enumerate(group.itertuples()):
            safe_label(row.strategy)
            frame = pd.read_csv(source / f"{asset}_{row.strategy}_{row.seed}_oos.csv",
                                parse_dates=["decision_time", "outcome_end"])
            if number == 0:
                arrays.update(decision_time=frame.decision_time.to_numpy(dtype="datetime64[ns]").astype("int64"),
                              outcome_end=frame.outcome_end.to_numpy(dtype="datetime64[ns]").astype("int64"),
                              asset_return=frame.asset_return.to_numpy(), fold=frame.fold.to_numpy())
            else:
                np.testing.assert_array_equal(arrays["decision_time"], frame.decision_time.to_numpy(dtype="datetime64[ns]").astype("int64"))
                np.testing.assert_allclose(arrays["asset_return"], frame.asset_return, atol=1e-15)
                np.testing.assert_array_equal(arrays["fold"], frame.fold)
            key = f"positions_{row.strategy}_{row.seed}"
            arrays[key] = frame.position.to_numpy()
            records.append({"strategy": row.strategy, "seed": int(row.seed), "key": key})
        metadata = {"version": 1, "asset": asset, "records": records,
                    "annualization": config["annualization"],
                    "cost_rate": (config["transaction_cost_bps"]+config["slippage_bps"])/10000}
        arrays["metadata"] = np.asarray(json.dumps(metadata))
        buffer = BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        path = source / f"history_{asset}.npz"
        if path.exists():
            raise FileExistsError(f"Archive already exists: {path.name}")
        if len(payload) > 2_000_000:
            raise ValueError("Archive exceeds the small-public-artifact limit")
        path.write_bytes(payload)
        checksums[path.name] = hashlib.sha256(payload).hexdigest()
        archives.append(path)
    index.write_text(json.dumps({"format": 1, "files": checksums,
                                 "contains": "Saved positions and realized return outcomes; no downloaded price bars or model weights."}, indent=2))
    return archives


def restore(source: Path) -> int:
    """Reconstruct missing CSVs and check saved metrics; never replace an existing CSV."""
    index = json.loads((source / "history_archives.json").read_text())
    raw = pd.read_csv(source / "metrics_by_seed.csv")
    created = 0
    for name, expected in index["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid archive filename")
        path = source / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Archive checksum mismatch")
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            asset = safe_label(metadata["asset"])
            dates = pd.DatetimeIndex(saved["decision_time"].astype("datetime64[ns]"))
            ends = pd.Series(saved["outcome_end"].astype("datetime64[ns]"), index=dates)
            returns = pd.Series(saved["asset_return"], index=dates)
            for record in metadata["records"]:
                strategy = safe_label(record["strategy"])
                seed = int(record["seed"])
                frame = account(saved[record["key"]], returns, ends, metadata["cost_rate"], liquidate=True)
                frame["fold"] = saved["fold"]
                frame.index.name = "decision_time"
                expected_row = raw[(raw.asset == asset) & (raw.strategy == strategy) & (raw.seed == seed)]
                if len(expected_row) != 1:
                    raise ValueError("Archive result has no unique metric row")
                for metric, actual in metrics(frame, metadata["annualization"]).items():
                    np.testing.assert_allclose(expected_row.iloc[0][metric], actual, atol=1e-10, rtol=1e-10)
                target = source / f"{asset}_{strategy}_{seed}_oos.csv"
                if target.exists():
                    old = pd.read_csv(target, parse_dates=["decision_time"]).set_index("decision_time")
                    if not old.index.equals(frame.index):
                        raise FileExistsError("Existing history has different dates; refuse overwrite")
                    for column in ["position", "asset_return", "net_return", "cost", "turnover", "fold", "liquidation_turnover"]:
                        np.testing.assert_allclose(old[column], frame[column], atol=1e-12, rtol=1e-12)
                else:
                    frame.to_csv(target)
                    created += 1
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.export:
        print([p.as_posix() for p in export(args.source)])
    else:
        print(f"Restored {restore(args.source)} missing result histories")
