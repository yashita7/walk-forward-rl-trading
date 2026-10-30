"""Single-asset continuous exposure environment sharing backtest accounting."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def net_step(position: float, previous: float, asset_return: float,
             cost_rate: float) -> tuple[float, float, float]:
    """One-way turnover; moving -1 to +1 costs twice the configured rate."""
    turnover = abs(position - previous)
    cost = cost_rate * turnover
    return position * asset_return - cost, turnover, cost


class TradingEnv(gym.Env):
    """Position at open t earns next open return; risk penalty affects reward only.

    Features contain one extra next-state row and optional left context;
    only [start, stop) earns rewards.
    The finite training horizon is a truncation, not an economic terminal state.
    """
    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, returns: np.ndarray, start: int,
                 stop: int, window: int, cost_rate: float,
                 volatility_penalty: float = 0.01, risk_window: int = 20,
                 turnover_penalty: float = 0.0, action_low: float = -1.0,
                 action_high: float = 1.0):
        super().__init__()
        self.features = np.asarray(features, dtype=np.float32)
        self.returns = np.asarray(returns, dtype=float)
        if not (window >= 1 and window - 1 <= start < stop <= len(returns)):
            raise ValueError("Invalid episode bounds or insufficient observation context")
        if len(features) != len(returns) + 1 or not np.isfinite(self.features).all() or not np.isfinite(self.returns).all():
            raise ValueError("Need one extra feature row for final next observation; all values must be finite")
        if cost_rate < 0 or volatility_penalty < 0 or risk_window < 2:
            raise ValueError("Costs/penalty must be nonnegative and risk_window >= 2")
        if not np.isfinite(turnover_penalty) or turnover_penalty < 0:
            raise ValueError("Turnover penalty must be finite and nonnegative")
        if not -1 <= action_low < action_high <= 1:
            raise ValueError("Action bounds must satisfy -1 <= low < high <= 1")
        self.start, self.stop, self.window = start, stop, window
        self.cost_rate, self.penalty, self.risk_window = cost_rate, volatility_penalty, risk_window
        self.turnover_penalty = turnover_penalty
        self.action_space = spaces.Box(action_low, action_high, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            (window * self.features.shape[1] + 1,), dtype=np.float32)
        self.reset()

    def _observation(self) -> np.ndarray:
        i = self.i
        return np.concatenate([self.features[i-self.window+1:i+1].ravel(),
                               [self.position]]).astype(np.float32)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset training exposure to cash, charging the first subsequent entry."""
        super().reset(seed=seed)
        self.i, self.position = self.start, 0.0
        self.history: list[float] = []
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        """Report net P&L; reward also subtracts optional risk/turnover penalties."""
        if self.i >= self.stop:
            raise RuntimeError("Episode ended; reset before stepping")
        position = float(np.clip(np.asarray(action).item(), self.action_space.low[0], self.action_space.high[0]))
        if not np.isfinite(position):
            raise ValueError("Action must be finite")
        net, turnover, cost = net_step(position, self.position, self.returns[self.i], self.cost_rate)
        self.history.append(net)
        risk = float(np.std(self.history[-self.risk_window:], ddof=0))
        self.position = position
        self.i += 1
        reward = net - self.penalty * risk - self.turnover_penalty * turnover
        return self._observation(), float(reward), False, self.i >= self.stop, {
            "net_return": net, "turnover": turnover, "cost": cost, "position": position,
        }
