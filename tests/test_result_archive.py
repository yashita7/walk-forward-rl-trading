"""Public archives reproduce results, detect corruption, and preserve existing files."""
import json

import numpy as np
import pandas as pd
import pytest

from src.backtest import account, metrics
from src.result_archive import export, restore


def fixture_results(tmp_path):
    dates = pd.date_range("2020-01-01", periods=5)
    rows, files = [], []
    for seed in [7, 21]:
        frame = account(np.array([.2, -.5, .1, .3]), pd.Series([.01, -.02, .03, -.01], index=dates[:4]),
                        pd.Series(dates[1:], index=dates[:4]), .0007, liquidate=True)
        frame["fold"] = [0, 0, 1, 1]
        frame.index.name = "decision_time"
        path = tmp_path / f"FIXTURE_ppo_{seed}_oos.csv"
        frame.to_csv(path)
        files.append(path)
        rows.append({"asset": "FIXTURE", "strategy": "ppo", "seed": seed, **metrics(frame)})
    pd.DataFrame(rows).to_csv(tmp_path / "metrics_by_seed.csv", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"annualization": 252,
        "transaction_cost_bps": 5, "slippage_bps": 2}}))
    return files


def test_archive_restores_missing_histories_and_never_overwrites_existing(tmp_path):
    files = fixture_results(tmp_path)
    original = pd.read_csv(files[0])
    archives = export(tmp_path)
    assert len(archives) == 1 and archives[0].stat().st_size < 2_000_000
    files[0].unlink()  # Isolated test fixture, never a research output.
    assert restore(tmp_path) == 1
    pd.testing.assert_frame_equal(original, pd.read_csv(files[0]))
    before = files[1].read_bytes()
    assert restore(tmp_path) == 0 and files[1].read_bytes() == before
    with pytest.raises(FileExistsError):
        export(tmp_path)


def test_archive_checksum_detects_corruption(tmp_path):
    fixture_results(tmp_path)
    path = export(tmp_path)[0]
    path.write_bytes(path.read_bytes()+b"damaged fixture")
    with pytest.raises(ValueError, match="checksum"):
        restore(tmp_path)
