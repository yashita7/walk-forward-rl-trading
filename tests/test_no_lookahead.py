"""Perturbation and fold-boundary checks guard causal information flow."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.backtest import walk_forward
from src.data import align_bars, synthetic_bars
from src.features import decision_frame, make_features


def test_future_mutation_cannot_change_past_features():
    bars = synthetic_bars(1)
    changed = bars.copy()
    changed.iloc[500:, :4] *= 10
    assert_frame_equal(make_features(bars).iloc[:500], make_features(changed).iloc[:500])
    original, _, _ = decision_frame(bars)
    mutated, _, _ = decision_frame(changed)
    assert_frame_equal(original.loc[:bars.index[500]], mutated.loc[:bars.index[500]])


def test_prefix_invariance():
    bars = synthetic_bars(1)
    assert_frame_equal(make_features(bars).iloc[:700], make_features(bars.iloc[:700]))


def test_next_open_timing():
    bars = synthetic_bars(2)
    x, y, ends = decision_frame(bars)
    date = x.index[20]
    i = bars.index.get_loc(date)
    np.testing.assert_allclose(x.loc[date], make_features(bars).iloc[i-1])
    assert y.loc[date] == bars.open.iloc[i+1] / bars.open.iloc[i] - 1
    assert ends.loc[date] == bars.index[i+1]


def test_fold_labels_purged_and_oos_nonoverlapping():
    x, _, ends = decision_frame(synthetic_bars(3, 2000))
    folds = list(walk_forward(x.index, ends, 3, 6, 20))
    assert len(folds) > 2
    all_test = []
    for fold in folds:
        assert (ends.iloc[fold.train] < fold.test_start).all()
        assert x.index[fold.train].max() < x.index[fold.test].min()
        assert x.index[fold.train].min() >= fold.test_start - pd.DateOffset(years=3)
        all_test.extend(fold.test)
    assert len(all_test) == len(set(all_test))
    assert np.all(np.diff(all_test) == 1)


def test_alignment_never_fills_missing_sessions():
    a = synthetic_bars(1, 10)
    b = a.drop(a.index[3])
    aligned = align_bars({"a": a, "b": b})
    assert a.index[3] not in aligned["a"].index
    assert aligned["a"].index.equals(aligned["b"].index)


def test_rl_train_only_scaling_and_future_test_invariance(monkeypatch):
    """Test prices/labels must not affect training or the first OOS action."""
    from sklearn.preprocessing import StandardScaler
    from src import train_rl

    fitted = []

    class AuditedScaler(StandardScaler):
        def fit(self, values, y=None, sample_weight=None):
            fitted.append(values.copy())
            return super().fit(values, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(train_rl, "StandardScaler", AuditedScaler)
    x, y, ends = decision_frame(synthetic_bars(4, 1100))
    fold = next(walk_forward(x.index, ends, 3, 6, 5))
    config = {"observation_window": 5, "transaction_cost_bps": 5,
              "slippage_bps": 2, "volatility_penalty": 0.01, "risk_window": 20,
              "ppo_n_steps": 32, "ppo_batch_size": 16, "ppo_timesteps": 32}
    before = train_rl.train_predict(x, y, fold, config, 7, 0)
    altered_x, altered_y = x.copy(), y.copy()
    altered_x.iloc[fold.test[0]+1:] += 100
    altered_y.iloc[fold.test] = 0.8
    after = train_rl.train_predict(altered_x, altered_y, fold, config, 7, 0)
    assert before[0] == after[0]
    assert len(fitted) == 2
    for fit_rows in fitted:
        assert_frame_equal(fit_rows, x.iloc[fold.train])
