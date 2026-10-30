"""Scientific checks must not alter test labels or conceal trading costs."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.backtest import account
from src.checks.common import shuffled_training_returns, summarize
from src.checks.cost_sensitivity import reprice
from src.checks.random_policy import random_positions


def test_shuffling_is_reproducible_preserves_distribution_and_leaves_test_untouched():
    y = pd.Series(np.arange(30) / 100, index=pd.date_range("2020-01-01", periods=30))
    train = np.arange(3, 20)
    original = y.copy()
    changed = shuffled_training_returns(y, train, 7)
    assert_series_equal(y, original)
    assert_series_equal(changed, shuffled_training_returns(y, train, 7))
    assert_series_equal(changed.iloc[20:], y.iloc[20:])
    assert_series_equal(changed.iloc[:3], y.iloc[:3])
    np.testing.assert_array_equal(np.sort(changed.iloc[train]), np.sort(y.iloc[train]))
    assert not changed.iloc[train].equals(y.iloc[train])


def test_cost_repricing_preserves_gross_return_and_exit_cost():
    dates = pd.date_range("2020-01-01", periods=4)
    y = pd.Series([.01, -.02, .03], index=dates[:3])
    ends = pd.Series(dates[1:], index=dates[:3])
    frame = account(np.array([1., -1., -1.]), y, ends, .0007, liquidate=True)
    pd.testing.assert_frame_equal(reprice(frame, 7), frame)
    free, costly = reprice(frame, 0), reprice(frame, 20)
    np.testing.assert_allclose(free.net_return - costly.net_return, .002 * frame.turnover)
    assert costly.cost.iloc[-1] == pytest.approx(.002)
    assert costly.turnover.sum() == 4
    with pytest.raises(ValueError):
        reprice(frame, -1)


def test_random_policy_matches_expected_turnover_and_is_symmetric():
    positions = random_positions(100000, .3, .2, 7)
    np.testing.assert_array_equal(positions, random_positions(100000, .3, .2, 7))
    assert np.abs(positions).mean() == pytest.approx(.3)
    assert abs(positions.mean()) < .01
    assert np.abs(np.diff(positions)).mean() == pytest.approx(2 * .3 * .2, abs=.003)
    np.testing.assert_array_equal(random_positions(10, 0, .2, 7), np.zeros(10))


def test_metrics_summary_is_across_seeds_not_averaged_returns():
    columns = ["annualized_return", "annualized_volatility", "sharpe", "max_drawdown", "turnover", "hit_rate"]
    raw = pd.DataFrame([{**{c: value for c in columns}, "asset": "X", "strategy": "ppo", "seed": seed}
                        for seed, value in [(7, 1.), (21, 3.)]])
    row = summarize(raw).iloc[0]
    assert row.sharpe_mean == 2
    assert row.sharpe_std == pytest.approx(np.sqrt(2))
    assert row.seed_count == 2


def test_experiment_shuffles_training_only_and_resumes_checkpoints(tmp_path, monkeypatch):
    from src.checks import common
    from src.data import synthetic_bars
    from src.features import decision_frame

    bars = synthetic_bars(9, 650)
    x, y, _ = decision_frame(bars)
    seen = []

    def fake_training(features, training_y, fold, config, seed, previous):
        pd.testing.assert_frame_equal(features, x)
        assert not training_y.iloc[fold.train].equals(y.iloc[fold.train])
        assert_series_equal(training_y.iloc[fold.test], y.iloc[fold.test])
        seen.append((fold.number, seed))
        return np.full(len(fold.test), .1 * seed)

    monkeypatch.setattr(common, "load_data", lambda config: {"X": bars})
    monkeypatch.setattr(common, "train_predict", fake_training)
    monkeypatch.setattr(common, "plot_results", lambda *args: None)
    config = {"train_years": 1, "test_months": 6, "observation_window": 5,
              "forecast_seed": 123, "seeds": [1, 2], "transaction_cost_bps": 5,
              "slippage_bps": 2, "annualization": 252}
    first = common.evaluate(config, tmp_path, shuffle=True)
    count = len(seen)
    assert count >= 2
    second = common.evaluate(config, tmp_path, shuffle=True)
    pd.testing.assert_frame_equal(first, second)
    assert len(seen) == count  # Identical run reused predictions, not stale/new fits.
    trace = pd.read_csv(tmp_path / "X_ppo_1_oos.csv")
    assert trace.liquidation_turnover.sum() == pytest.approx(.1)
    assert trace.turnover.sum() == pytest.approx(.2)
    assert len(trace) == trace.decision_time.nunique()


def test_full_run_requires_current_estimate_and_long_run_decision():
    from src.checks.run_final import validate_runtime

    config = {"ppo_timesteps": 10000, "seeds": [7, 21, 42, 84, 123]}
    estimate = {"config": config, "more_than_30_minutes": False}
    validate_runtime(estimate, config, False)
    estimate["more_than_30_minutes"] = True
    with pytest.raises(ValueError, match="exceeds 30"):
        validate_runtime(estimate, config, False)
    validate_runtime(estimate, config, True)
    with pytest.raises(ValueError, match="changed"):
        validate_runtime(estimate, {"ppo_timesteps": 256}, True)
