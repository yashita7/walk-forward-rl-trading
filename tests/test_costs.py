import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from src.backtest import account
from src.env import TradingEnv


def toy():
    dates = pd.date_range("2020-01-01", periods=4)
    return pd.Series([0.01, -0.02, 0.03], index=dates[:3]), pd.Series(dates[1:], index=dates[:3])


def test_entry_flip_exit_costs_and_env_parity():
    returns, ends = toy()
    frame = account(np.array([1, -1, 0]), returns, ends, 0.0007)
    np.testing.assert_allclose(frame.turnover, [1, 2, 1])
    np.testing.assert_allclose(frame.cost, [0.0007, 0.0014, 0.0007])
    np.testing.assert_allclose(frame.net_return, [0.0093, 0.0186, -0.0007])
    env = TradingEnv(np.zeros((4, 2)), returns.to_numpy(), 0, 3, 1, 0.0007, 0)
    env.reset()
    for i, action in enumerate([1, -1, 0]):
        _, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        assert reward == pytest.approx(frame.net_return.iloc[i])
        assert info["cost"] == pytest.approx(frame.cost.iloc[i])
        assert not terminated
        assert truncated == (i == 2)
    with pytest.raises(RuntimeError):
        env.step(np.array([0]))


def test_fold_boundary_does_not_reset_exposure():
    returns, ends = toy()
    first = account(np.array([1]), returns.iloc[:1], ends.iloc[:1], 0.0007)
    rest = account(np.array([1, -1]), returns.iloc[1:], ends.iloc[1:], 0.0007, first.position.iloc[-1])
    np.testing.assert_allclose(rest.cost, [0, 0.0014])


def test_risk_penalty_does_not_contaminate_reported_returns():
    env = TradingEnv(np.zeros((4, 1)), np.array([0.01, -0.01, 0]), 0, 3, 1, 0, 0.1)
    env.step(np.array([1]))
    _, reward, _, _, info = env.step(np.array([1]))
    assert info["net_return"] == -0.01
    assert reward == pytest.approx(-0.011)


def test_final_liquidation_is_charged_once():
    returns, ends = toy()
    frame = account(np.array([1, 1, 1]), returns, ends, 0.0007, liquidate=True)
    np.testing.assert_allclose(frame.cost, [0.0007, 0, 0.0007])
    assert frame.net_return.iloc[-1] == pytest.approx(0.0293)
    assert frame.liquidation_turnover.sum() == 1


def test_gymnasium_api():
    env = TradingEnv(np.zeros((7, 2)), np.zeros(6), 2, 6, 3, 0.0007)
    check_env(env, skip_render_check=True)
