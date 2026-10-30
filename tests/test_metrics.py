import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics


def test_hand_computed_metrics():
    # NAV 1 -> 1.1 -> .99 -> 1.089; sample std sqrt(.04/3).
    frame = pd.DataFrame({"net_return": [0.1, -0.1, 0.1], "position": [1, 1, 1], "turnover": [1, 0, 0]})
    result = metrics(frame, annualization=3)
    std = np.sqrt(0.04 / 3)
    assert result["annualized_return"] == pytest.approx(0.089)
    assert result["annualized_volatility"] == pytest.approx(std * np.sqrt(3))
    assert result["sharpe"] == pytest.approx((0.1 / 3) / std * np.sqrt(3))
    assert result["max_drawdown"] == pytest.approx(-0.1)
    assert result["turnover"] == pytest.approx(1 / 3)
    assert result["hit_rate"] == pytest.approx(2 / 3)


def test_first_loss_counts_in_drawdown_and_cash_is_excluded_from_hits():
    frame = pd.DataFrame({"net_return": [-0.1, 0, 0], "position": [1, 0, 0], "turnover": [1, 1, 0]})
    assert metrics(frame)["max_drawdown"] == pytest.approx(-0.1)
    assert metrics(frame)["hit_rate"] == 0


def test_zero_volatility_and_bankruptcy():
    frame = pd.DataFrame({"net_return": [0, 0], "position": [0, 0], "turnover": [0, 0]})
    assert all(value == 0 for value in metrics(frame).values())
    frame.loc[0, "net_return"] = -1
    with pytest.raises(ValueError, match="bankruptcy"):
        metrics(frame)
