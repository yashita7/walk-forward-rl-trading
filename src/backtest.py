"""Purged calendar walk-forward splits, common accounting, and net OOS metrics."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np
import pandas as pd

from .env import net_step


@dataclass(frozen=True)
class Fold:
    number: int
    train: np.ndarray
    test: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward(index: pd.DatetimeIndex, outcome_end: pd.Series,
                 train_years: int, test_months: int, window: int) -> Iterator[Fold]:
    """Nonoverlapping test blocks; purge labels unavailable before test open.

    Training feature/context rows all lie in the rolling training interval.
    Test observations can use historical context before the test interval.
    """
    if train_years < 1 or test_months < 1 or window < 1:
        raise ValueError("Window lengths must be positive")
    boundary = index[0] + pd.DateOffset(years=train_years)
    number = 0
    while boundary <= index[-1]:
        end = boundary + pd.DateOffset(months=test_months)
        train_start = boundary - pd.DateOffset(years=train_years)
        train = np.flatnonzero((index >= train_start) & (index < boundary) &
                               (outcome_end.to_numpy() < boundary.to_datetime64()))
        test = np.flatnonzero((index >= boundary) & (index < end))
        if len(train) >= window + 2 and len(test) and test[0] >= window - 1:
            yield Fold(number, train, test, boundary, end)
            number += 1
        boundary = end


def account(positions: np.ndarray, returns: pd.Series, outcome_end: pd.Series,
            cost_rate: float, previous: float = 0.0, liquidate: bool = False) -> pd.DataFrame:
    """Account for each decision, including first entry and optional final exit.

    Position means target fraction of NAV. Daily rebalancing/financing are
    simplified: turnover is the requested target change, not drift turnover.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) != len(returns) or not np.isfinite(positions).all() or (abs(positions) > 1).any():
        raise ValueError("Positions must be aligned, finite, and in [-1, 1]")
    if cost_rate < 0:
        raise ValueError("Cost rate must be nonnegative")
    rows = []
    for position, asset_return in zip(positions, returns):
        net, turnover, cost = net_step(position, previous, float(asset_return), cost_rate)
        rows.append((position, asset_return, turnover, cost, net))
        previous = position
    result = pd.DataFrame(rows, index=returns.index,
                          columns=["position", "asset_return", "turnover", "cost", "net_return"])
    result["outcome_end"] = outcome_end
    result["liquidation_turnover"] = 0.0
    if liquidate and len(result):
        exit_turnover = abs(previous)
        # Exit at final outcome open; all cost fractions use start-of-period NAV.
        last = result.index[-1]
        result.loc[last, "liquidation_turnover"] = exit_turnover
        result.loc[last, "turnover"] += exit_turnover
        result.loc[last, "cost"] += cost_rate * exit_turnover
        result.loc[last, "net_return"] -= cost_rate * exit_turnover
    return result


def equity_and_drawdown(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Include initial NAV=1 in the high-water mark, including first-day loss."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1)
    return equity, equity / peak - 1


def metrics(frame: pd.DataFrame, annualization: int = 252) -> dict[str, float]:
    """Geometric annual return; sample vol; zero-rate Sharpe; average daily turnover.

    Hit rate is fraction of exposed periods with strictly positive net P&L.
    Flat periods are excluded. No active observations => hit rate zero.
    """
    if frame.empty or annualization <= 0:
        raise ValueError("Metrics require observations and positive annualization")
    r = frame.net_return.astype(float)
    if not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Nonfinite returns or NAV bankruptcy; metrics undefined")
    equity, drawdown = equity_and_drawdown(r)
    vol = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    active = frame.position.abs() > 1e-8
    return {
        "annualized_return": float(equity.iloc[-1] ** (annualization / len(r)) - 1),
        "annualized_volatility": vol * np.sqrt(annualization),
        "sharpe": float(r.mean() / vol * np.sqrt(annualization)) if vol > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(frame.turnover.mean()),
        "hit_rate": float((r[active] > 0).mean()) if active.any() else 0.0,
    }
