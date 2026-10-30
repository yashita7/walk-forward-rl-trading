# Robustness checks: what happened and what it means

These are diagnostics, meaning deliberate tests of the experiment. They do not
choose trading settings. All use the same real test returns, all 17 chronological
folds (training/test pairs) per asset, and five fixed random seeds (repeatable random
starting points). Ordinary and shuffled training each receive
1,024 PPO learning steps per fold. This is less than
the 10,000-step final budget; differences in the final run must not be attributed
solely to shuffling. Net means after the default 7 bps all-in trading cost; one
basis point (bp) is 0.01%. Values following ± are sample standard deviations,
meaning the spread across the five runs, not confidence intervals.

## 1. Shuffled training answers

A label is the future return used as the answer during training. We randomly
reorder only each fold's training labels, independently for each asset/fold/seed.
The forecasting model learns from reordered labels; PPO receives those reordered
returns when calculating its training rewards. Features, dates, and actual test
answers stay unchanged. The ordinary comparison uses identical dates, seeds, and
learning budgets. Buy-and-hold and MA rules ignore training labels and do not change.

| Asset | Strategy | Ordinary net annual return | Shuffled net annual return | Ordinary Sharpe | Shuffled Sharpe |
| --- | --- | --- | --- | --- | --- |
| BTC-USD | gradient_boosting | 24.78% ± 0.00% | -6.94% ± 17.89% | 0.66 ± 0.00 | 0.17 ± 0.31 |
| BTC-USD | ppo | -1.70% ± 3.13% | -1.68% ± 3.69% | -0.18 ± 0.44 | -0.17 ± 0.48 |
| QQQ | gradient_boosting | 6.40% ± 0.00% | -7.37% ± 5.66% | 0.38 ± 0.00 | -0.21 ± 0.26 |
| QQQ | ppo | -0.88% ± 0.76% | -0.85% ± 0.72% | -0.26 ± 0.22 | -0.24 ± 0.20 |
| SPY | gradient_boosting | 0.23% ± 0.00% | -8.38% ± 5.37% | 0.11 ± 0.00 | -0.39 ± 0.31 |
| SPY | ppo | -1.23% ± 0.73% | -1.25% ± 0.81% | -0.53 ± 0.32 | -0.53 ± 0.35 |

In this run, PPO's average Sharpe barely changed when its training answers were
scrambled (BTC-USD: -0.18 ordinary versus -0.17 shuffled; QQQ: -0.26 ordinary versus -0.24 shuffled; SPY: -0.53 ordinary versus -0.53 shuffled). At this budget, that provides little
evidence that PPO learned useful timing of daily returns. The forecasting model's
average return fell after shuffling on all three assets, but this comparison alone
does not establish reliable forecasting skill or superiority to buy-and-hold.
The full-budget PPO run is a different experiment, so this smaller-budget result
cannot by itself rule out anything learned with more training.

**Why is the shuffled result not necessarily zero?** Shuffling breaks the match
between inputs and answers, but preserves the average training return. An agent
can learn to stay long, meaning hold exposure that gains when prices rise, even
without predicting which days rise. The real test market can rise or fall; a
constant or lucky exposure then makes or loses money. Trading costs push active
zero-signal policies below zero. Compounding, repeatedly applying returns to
changing wealth, also makes geometric growth differ from average daily profit.
Therefore nonzero shuffled returns alone are neither proof of a software bug nor
proof of forecasting skill. Compare ordinary vs shuffled at the same budget, inspect
market exposure and the random-policy distribution, and investigate unexpectedly
strong persistent differences. This is five random permutations per fold, not a
formal statistical significance test or an acceptance rule requiring zero profit.

## 2. Cost sensitivity

We keep learned decisions fixed and rerun the accounting at 0, 5, and 20 bps total
one-way cost, also showing the actual default of 7 bps = 5 fee + 2 slippage.
Slippage means receiving a worse trading price than assumed. Zero here truly means
zero fees AND zero slippage. Repricing includes first entries, reversals, and final
exits. This isolates the effect of fees; it does not train a different PPO policy
at each fee. Sharpe is average net return divided by return variability, scaled
to a year. Higher is usually preferable, but it need not fall monotonically with
cost because cost can also change the denominator, the variability.

| Asset | Strategy | Sharpe: 0 bps | 5 bps | 7 bps | 20 bps |
| --- | --- | --- | --- | --- | --- |
| BTC-USD | buy_hold | 0.83 | 0.83 | 0.83 | 0.83 |
| BTC-USD | gradient_boosting | 0.77 | 0.70 | 0.66 | 0.46 |
| BTC-USD | ma_50_200 | 0.84 | 0.83 | 0.83 | 0.83 |
| BTC-USD | ppo | -0.01 | -0.13 | -0.18 | -0.49 |
| QQQ | buy_hold | 0.90 | 0.90 | 0.90 | 0.89 |
| QQQ | gradient_boosting | 0.59 | 0.44 | 0.38 | -0.00 |
| QQQ | ma_50_200 | 0.91 | 0.90 | 0.90 | 0.89 |
| QQQ | ppo | 0.18 | -0.13 | -0.26 | -1.08 |
| SPY | buy_hold | 0.86 | 0.86 | 0.86 | 0.86 |
| SPY | gradient_boosting | 0.40 | 0.19 | 0.11 | -0.44 |
| SPY | ma_50_200 | 0.67 | 0.66 | 0.66 | 0.65 |
| SPY | ppo | 0.06 | -0.36 | -0.53 | -1.62 |

## 3. What luck looks like

For each asset, 1,000 random policies flip between equally sized
long and short positions. Average absolute exposure, meaning how much money is at
risk regardless of direction, is matched to the ordinary PPO seeds. The probability
of flipping direction is chosen to match average internal turnover (position changes
between days); initial entry and final exit are also charged. Actual matching error
is reported below. Matching uses observed test positions, not test returns, so
this is a retrospective comparison and is not a deployable calibrated strategy.

| Asset | Target exposure | Target daily internal turnover | Random actual turnover | Random middle 90% Sharpe | PPO percentile range |
| --- | --- | --- | --- | --- | --- |
| BTC-USD | 0.1033 | 0.0694 | 0.0694 | -0.74 to 0.37 | 8.8–94.7 |
| QQQ | 0.1144 | 0.0795 | 0.0796 | -1.08 to 0.01 | 53.6–96.5 |
| SPY | 0.1082 | 0.0763 | 0.0763 | -1.22 to -0.09 | 29.8–93.9 |

A percentile of 80 means that PPO seed's Sharpe exceeded about 80% of these random
runs. It is not an 80% chance of future success or a formal probability of skill.
The middle 90% interval describes outcomes on this one realized market history;
it is not uncertainty across possible future markets. Matching only average
exposure and turnover does not reproduce PPO's entire action distribution.

## Files and reproduction

Run `python -m src.checks.run_checks`. `plan.json` freezes the settings before
training. `ordinary/` and `shuffled/` contain per-seed metrics, per-day accounting,
fold bounds, timing, and manifests. `cost_sensitivity/` contains four-cost metrics
and Sharpe plots. `random_policy/` contains every random run, matching details,
seed percentiles, and histograms. Prediction checkpoints can resume identical
experiments; source/config/data signatures prevent reuse after an input change.
No result was used to change features, seeds, costs, or model parameters.

Shuffling also destroys the original order of returns and makes simulated reward
history inconsistent with the lagged real-return features. It is a deliberately
artificial control, not a realistic alternate market. Five permutations per fold
provide limited evidence; no formal significance claim is made.
