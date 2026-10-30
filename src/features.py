"""Causal close-bar features, delayed one session for executable open decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Features stamped t use only completed bars through t; no fitted transforms."""
    close = bars.close
    log_return = np.log(close).diff()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 * gain / (gain + loss)
    rsi = rsi.mask((gain + loss) == 0, 50)
    return pd.DataFrame({
        "log_return": log_return,
        "return_lag_1": log_return.shift(1),
        "return_lag_2": log_return.shift(2),
        "volatility_20": log_return.rolling(20).std(),
        "momentum_20": close.pct_change(20, fill_method=None),
        "rsi_14": rsi / 100,
        "ma_spread": close.rolling(50).mean() / close.rolling(200).mean() - 1,
    }, index=bars.index)


def decision_frame(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """At open t observe features from t-1; action earns open[t+1]/open[t]-1.

    Outcome/end timestamps are separate from predictors and never observations.
    """
    x = make_features(bars).shift(1)
    returns = bars.open.shift(-1) / bars.open - 1
    ends = pd.Series(bars.index, index=bars.index).shift(-1)
    valid = x.notna().all(axis=1) & returns.notna() & ends.notna()
    return x.loc[valid], returns.loc[valid].rename("asset_return"), ends.loc[valid]
