"""The four fixed hypotheses in docs/EXPERIMENT_PLAN.md; no parameter selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

VARIANTS = ("ppo_turnover_penalty", "ppo_long_only", "vol_target_buy_hold", "gb_fixed_features")


def ppo_configuration(config: dict, variant: str) -> dict:
    """Apply exactly one requested change while leaving monetary costs unchanged."""
    if variant not in VARIANTS[:2]:
        raise ValueError("Not a PPO variant")
    settings = config["improvement_round"]
    result = dict(config)
    result.update(ppo_settings=settings["ppo"], policy_settings=settings["policy"],
                  action_high=settings["action_high"],
                  action_low=settings["long_only_action_low"] if variant == "ppo_long_only" else settings["turnover_action_low"],
                  turnover_penalty=settings["turnover_penalty"] if variant == "ppo_turnover_penalty" else 0.0)
    return result


def volatility_target_positions(bars: pd.DataFrame, config: dict) -> pd.Series:
    """At open t size only from close returns available through t-1; no leverage."""
    settings = config["improvement_round"]
    historical = bars.close.pct_change(fill_method=None).rolling(settings["volatility_lookback"]).std(ddof=1).shift(1)
    annual_risk = historical * np.sqrt(config["annualization"])
    return (settings["volatility_target"] / annual_risk.clip(lower=settings["volatility_floor"])).clip(
        lower=0, upper=settings["maximum_position"])


def forecast_positions(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame,
                       config: dict, seed: int) -> np.ndarray:
    """Fixed five-feature robust boosting, with exposure scaled by training risk."""
    settings = config["improvement_round"]
    columns = settings["forecast_features"]
    model = GradientBoostingRegressor(random_state=seed, **settings["forecast"])
    model.fit(train_x[columns], train_y)
    scale = max(float(train_y.std(ddof=1)), settings["forecast_scale_floor"])
    prediction = model.predict(test_x[columns])
    return np.clip(prediction / (scale * settings["forecast_prediction_scale"]), -1., 1.)
