"""Fixed, predeclared baselines fitted only within each training fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def baseline_positions(train_x: pd.DataFrame, train_y: pd.Series,
                       test_x: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Predict next open-to-open return; sign maps forecasts to unit exposure."""
    model = GradientBoostingRegressor(n_estimators=75, max_depth=2,
                                      learning_rate=0.05, random_state=seed)
    model.fit(train_x, train_y)
    return {
        "buy_hold": np.ones(len(test_x)),
        "ma_50_200": (test_x.ma_spread > 0).to_numpy(dtype=float),
        "gradient_boosting": np.sign(model.predict(test_x)),
    }
