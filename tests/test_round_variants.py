"""Causality, accounting, and immutable experiment retention for the fixed round."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest import account, metrics, walk_forward
from src.data import synthetic_bars
from src.diagnostics import position_statistics
from src.env import TradingEnv
from src.features import decision_frame
from src.round_variants import forecast_positions, ppo_configuration, volatility_target_positions


@pytest.fixture
def config():
    return yaml.safe_load(Path("config.yaml").read_text())


def test_turnover_penalty_changes_reward_but_not_money():
    env = TradingEnv(np.zeros((4, 1)), np.array([.01, -.02, 0]), 0, 3, 1,
                     .0007, volatility_penalty=0, turnover_penalty=.001)
    _, reward, _, _, info = env.step(np.array([1.]))
    assert info["net_return"] == pytest.approx(.0093)
    assert info["cost"] == pytest.approx(.0007)
    assert reward == pytest.approx(.0083)
    _, reward, _, _, info = env.step(np.array([-1.]))
    assert info["net_return"] == pytest.approx(.0186)
    assert reward == pytest.approx(.0166)


def test_long_only_environment_clips_negative_actions():
    env = TradingEnv(np.zeros((3, 1)), np.array([.01, .02]), 0, 2, 1, .0007, action_low=0)
    _, _, _, _, info = env.step(np.array([-1.]))
    assert env.action_space.low.item() == 0
    assert info["position"] == 0 and info["net_return"] == 0
    with pytest.raises(ValueError, match="bounds"):
        TradingEnv(np.zeros((3, 1)), np.zeros(2), 0, 2, 1, .0007, action_low=1)


def test_ppo_variants_change_one_thing_and_preserve_real_costs(config):
    penalty = ppo_configuration(config, "ppo_turnover_penalty")
    long = ppo_configuration(config, "ppo_long_only")
    assert penalty["action_low"] == -1 and penalty["turnover_penalty"] == .001
    assert long["action_low"] == 0 and long["turnover_penalty"] == 0
    assert "turnover_penalty" not in config
    assert penalty["transaction_cost_bps"] == long["transaction_cost_bps"] == 5


def test_volatility_target_is_causal_and_matches_hand_calculation(config):
    bars = synthetic_bars(1, 80)
    before = volatility_target_positions(bars, config)
    modified = bars.copy()
    modified.iloc[50:, modified.columns.get_loc("close")] *= 10
    pd.testing.assert_series_equal(before.iloc[:51], volatility_target_positions(modified, config).iloc[:51])
    sigma = bars.close.pct_change(fill_method=None).iloc[30:50].std(ddof=1) * np.sqrt(252)
    assert before.iloc[50] == pytest.approx(min(1, .10 / max(sigma, .01)))
    assert before.dropna().between(0, 1).all()


def test_forecast_fits_fixed_columns_and_scales_only_from_training(config, monkeypatch):
    from src import round_variants
    columns = config["improvement_round"]["forecast_features"]
    train_x = pd.DataFrame({c: [1., 2., 3.] for c in columns + ["unused_future"]})
    test_x = train_x.iloc[:2].copy()
    y = pd.Series([-.02, 0, .02])
    seen = []

    class Model:
        def __init__(self, **kwargs):
            assert kwargs["n_estimators"] == 200
            assert kwargs["min_samples_leaf"] == 20
        def fit(self, x, labels):
            seen.append((x.copy(), labels.copy()))
        def predict(self, x):
            assert list(x.columns) == columns
            return np.array([.01, -.06])

    monkeypatch.setattr(round_variants, "GradientBoostingRegressor", Model)
    result = forecast_positions(train_x, y, test_x, config, 7)
    np.testing.assert_allclose(result, [.5, -1])
    pd.testing.assert_frame_equal(seen[0][0], train_x[columns])
    pd.testing.assert_series_equal(seen[0][1], y)


@pytest.mark.parametrize("variant", ["ppo_turnover_penalty", "ppo_long_only"])
def test_actual_ppo_accepts_frozen_policy_settings(config, variant):
    from src.train_rl import train_predict
    x, y, ends = decision_frame(synthetic_bars(8, 650))
    fold = next(walk_forward(x.index, ends, 1, 6, 5))
    fold = replace(fold, test=fold.test[:3])
    fixture = copy.deepcopy(config)
    fixture.update(observation_window=5, ppo_n_steps=32, ppo_batch_size=16, ppo_timesteps=32)
    values = train_predict(x, y, fold, ppo_configuration(fixture, variant), 7, 0)
    assert len(values) == 3 and np.isfinite(values).all()
    assert (values >= (0 if variant == "ppo_long_only" else -1)).all()
    assert (values <= 1).all()


def test_diagnostic_fractions_and_gross_net_calculation():
    dates = pd.date_range("2020-01-01", periods=4)
    frame = account(np.array([1., -1., 0]), pd.Series([.01, -.02, .03], index=dates[:3]),
                    pd.Series(dates[1:], index=dates[:3]), .0007, liquidate=True)
    result = position_statistics(frame, annualization=3)
    assert result["gross_annualized_return"] == pytest.approx(1.01 * 1.02 - 1)
    assert result["long_fraction"] == result["short_fraction"] == result["flat_fraction"] == pytest.approx(1/3)
    assert result["average_absolute_position"] == pytest.approx(2/3)
    assert result["net_annualized_return"] < result["gross_annualized_return"]


def test_dirty_plan_is_rejected_before_training(monkeypatch):
    from src import improvement_round
    monkeypatch.setattr(improvement_round, "git_read", lambda *args: " M config.yaml")
    with pytest.raises(ValueError, match="Commit"):
        improvement_round.committed_plan()


def test_original_reference_must_match_test_dates_and_prices():
    from src.improvement_round import audit_original
    dates = pd.date_range("2020-01-01", periods=3)
    values = pd.Series([.01, .02, .03], index=dates)
    frame = pd.DataFrame({"asset_return": values})
    audit_original(frame, dates, values)
    with pytest.raises(ValueError, match="dates differ"):
        audit_original(frame.iloc[1:], dates, values)
    frame.iloc[0, 0] = .99
    with pytest.raises(AssertionError):
        audit_original(frame, dates, values)


def test_incomplete_round_cannot_be_reported_as_completed(tmp_path):
    from src.improvement_report import write_report
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="incomplete/failed"):
        write_report(tmp_path)


def test_failed_round_keeps_errors_and_resumes_without_repeating_saved_fit(config, tmp_path, monkeypatch):
    from src import improvement_round
    fixture = copy.deepcopy(config)
    fixture.update(assets=["FIXTURE"], train_years=1, observation_window=5)
    fixture["improvement_round"]["original_results"] = "original"
    bars = synthetic_bars(5, 650)
    x, y, ends = decision_frame(bars)
    folds = list(walk_forward(x.index, ends, 1, 6, 5))
    indices = np.concatenate([fold.test for fold in folds])
    monkeypatch.chdir(tmp_path)
    Path("docs").mkdir()
    Path("docs/EXPERIMENT_PLAN.md").write_text("Fixed test fixture, not research")
    Path("config.yaml").write_text(yaml.safe_dump(fixture))
    original = Path("original")
    original.mkdir()
    (original / "manifest.json").write_text(json.dumps({"config": fixture}))
    rows = []
    for strategy in ["buy_hold", "ma_50_200", "gradient_boosting", "ppo"]:
        for seed in fixture["seeds"]:
            frame = account(np.ones(len(indices)), y.iloc[indices], ends.iloc[indices], .0007, liquidate=True)
            frame.index.name = "decision_time"
            frame.to_csv(original / f"FIXTURE_{strategy}_{seed}_oos.csv")
            rows.append({"asset": "FIXTURE", "strategy": strategy, "seed": seed, **metrics(frame)})
    pd.DataFrame(rows).to_csv(original / "metrics_by_seed.csv", index=False)
    monkeypatch.setattr(improvement_round, "committed_plan", lambda: {"plan_commit": "fixture", "code_commit": "fixture"})
    monkeypatch.setattr(improvement_round, "load_data", lambda cfg: {"FIXTURE": bars})
    calls = []
    failed = False

    def fake_training(features, targets, fold, cfg, seed, prior):
        nonlocal failed
        if cfg["action_low"] == 0 and not failed:
            failed = True
            raise RuntimeError("Deliberate synthetic fixture failure")
        calls.append((fold.number, seed, cfg["action_low"]))
        return np.full(len(fold.test), .2)

    monkeypatch.setattr(improvement_round, "train_predict", fake_training)
    monkeypatch.setattr(improvement_round, "forecast_positions", lambda tx, ty, vx, cfg, seed: np.full(len(vx), .1))
    output = Path("round")
    with pytest.raises(RuntimeError, match="Deliberate"):
        improvement_round.run(fixture, output)
    assert (output / "failure_001.json").exists()
    improvement_round.run(fixture, output, resume=True)
    assert len(calls) == 2 * len(folds) * 5
    assert len(pd.read_csv(output / "metrics.csv")) == 8
    assert len(pd.read_csv(output / "metrics_by_seed.csv")) == 40
    assert (output / "failure_001.json").exists()
    metadata = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        improvement_round.run(fixture, output, resume=True)
    assert (output / "manifest.json").read_bytes() == metadata
    assert "round_failed" in (output / "ledger.jsonl").read_text()
