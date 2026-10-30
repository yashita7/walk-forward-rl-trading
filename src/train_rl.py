"""Fresh CPU PPO and train-only standardization for each asset/fold/seed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from .backtest import Fold
from .env import TradingEnv


def train_predict(x: pd.DataFrame, y: pd.Series, fold: Fold, config: dict,
                  seed: int, previous: float) -> np.ndarray:
    """Fit only training features/labels, then deterministic sequential test inference."""
    torch.set_num_threads(1)
    train_x = x.iloc[fold.train]
    scaler = StandardScaler().fit(train_x)
    # Extra next-state row enables correct timeout bootstrapping; it is never fitted.
    train_context = x.iloc[fold.train[0]:fold.train[-1]+2]
    train_features = scaler.transform(train_context).astype(np.float32)
    window = config["observation_window"]
    env = TradingEnv(train_features, y.iloc[fold.train].to_numpy(), window - 1,
                     len(train_x), window,
                     (config["transaction_cost_bps"] + config["slippage_bps"]) / 10000,
                     config["volatility_penalty"], config["risk_window"],
                     config.get("turnover_penalty", 0.0), config.get("action_low", -1.0),
                     config.get("action_high", 1.0))
    policy = dict(config.get("policy_settings", {"net_arch": [32, 32]}))
    if "activation" in policy:
        if policy.pop("activation") != "Tanh" or policy.pop("optimizer") != "Adam":
            raise ValueError("This fixed round supports only Tanh and Adam")
        policy["activation_fn"] = torch.nn.Tanh
        policy["optimizer_class"] = torch.optim.Adam
        policy["optimizer_kwargs"] = {"eps": policy.pop("optimizer_epsilon")}
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                n_steps=config["ppo_n_steps"], batch_size=config["ppo_batch_size"],
                policy_kwargs=policy, **config.get("ppo_settings", {}))
    model.learn(total_timesteps=config["ppo_timesteps"])
    # Transform only the causal context plus this test block; no test fitting.
    left = fold.test[0] - window + 1
    test_features = scaler.transform(x.iloc[left:fold.test[-1]+1]).astype(np.float32)
    positions = []
    for j in range(len(fold.test)):
        observation = np.concatenate([test_features[j:j+window].ravel(), [previous]]).astype(np.float32)
        action, _ = model.predict(observation, deterministic=True)
        previous = float(np.clip(action.item(), env.action_space.low[0], env.action_space.high[0]))
        positions.append(previous)
    env.close()
    return np.asarray(positions)
