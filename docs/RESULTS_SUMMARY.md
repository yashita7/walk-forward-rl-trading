# Results summary: full-budget experiment

## In plain language

We repeatedly learned from earlier three-year windows and tested on the following up-to-six months, using 17 test windows per asset, five random starting points, and a trading charge of 7 basis points (0.07%) per unit of exposure changed.
PPO performed worse than every baseline on every asset by both average annual return after costs and Sharpe, which measures average return relative to its variability.
The numbers of PPO runs with positive annual returns were BTC-USD: 1 of 5; QQQ: 0 of 5; SPY: 0 of 5, so losses were consistent across starting points for SPY and QQQ.
The ± values measure variation across starting points rather than uncertainty about future markets, while the simple rules have zero variation because they are repeated unchanged.
Keeping decisions fixed, increasing fees plus slippage from zero to 20 basis points worsened PPO's average Sharpe on all three assets, and even free trading left it behind buy-and-hold on this measure.
At the separate, equal 1,024-step training budget, scrambling the training answers barely changed PPO's results, providing little evidence that it learned useful daily return timing at that budget.
The 1,000 random-agent comparisons per asset showed that a PPO starting point could rank well against luck while still losing money, so one favorable rank does not establish reliable skill.
We did not tune using these test results, and revised data, asset selection, simplified trading assumptions, and future market changes still limit the conclusion.

## Results table

Net means after fees and slippage; bps means basis points, each equal to 0.01%.
Mean is the average across five seeds (repeatable random starting states), and
standard deviation after ± measures their spread. Volatility measures how variable
returns are; Sharpe divides average return by that variability and scales to a
year; max drawdown is the worst fall from a previous wealth peak; turnover measures
position changes; hit rate counts profitable exposed periods. Daily turnover is a
fraction of account exposure, so 1.0 means 100% of account value changed hands.

| Asset | Strategy | Annual net return | Annual volatility | Sharpe | Max drawdown | Daily turnover | Hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTC-USD | buy_hold | 37.16% ± 0.00% | 60.96% ± 0.00% | 0.83 ± 0.00 | -76.64% ± 0.00% | 0.0010 ± 0.0000 | 49.85% ± 0.00% |
| BTC-USD | gradient_boosting | 24.78% ± 0.00% | 60.96% ± 0.00% | 0.66 ± 0.00 | -77.74% ± 0.00% | 0.3760 ± 0.0000 | 49.90% ± 0.00% |
| BTC-USD | ma_50_200 | 32.27% ± 0.00% | 46.23% ± 0.00% | 0.83 ± 0.00 | -60.57% ± 0.00% | 0.0050 ± 0.0000 | 50.28% ± 0.00% |
| BTC-USD | ppo | -10.25% ± 9.42% | 43.52% ± 1.45% | -0.03 ± 0.23 | -84.49% ± 6.92% | 0.5277 ± 0.0200 | 48.22% ± 1.07% |
| QQQ | buy_hold | 20.03% ± 0.00% | 23.47% ± 0.00% | 0.90 ± 0.00 | -36.69% ± 0.00% | 0.0010 ± 0.0000 | 55.75% ± 0.00% |
| QQQ | gradient_boosting | 6.40% ± 0.00% | 23.52% ± 0.00% | 0.38 ± 0.00 | -44.51% ± 0.00% | 0.2768 ± 0.0000 | 52.58% ± 0.00% |
| QQQ | ma_50_200 | 17.44% ± 0.00% | 20.09% ± 0.00% | 0.90 ± 0.00 | -27.56% ± 0.00% | 0.0050 ± 0.0000 | 56.69% ± 0.00% |
| QQQ | ppo | -9.88% ± 1.47% | 15.95% ± 0.79% | -0.57 ± 0.07 | -61.29% ± 7.47% | 0.4878 ± 0.0206 | 45.69% ± 1.07% |
| SPY | buy_hold | 15.30% ± 0.00% | 18.46% ± 0.00% | 0.86 ± 0.00 | -32.05% ± 0.00% | 0.0010 ± 0.0000 | 55.85% ± 0.00% |
| SPY | gradient_boosting | 0.23% ± 0.00% | 18.49% ± 0.00% | 0.11 ± 0.00 | -39.18% ± 0.00% | 0.3085 ± 0.0000 | 51.69% ± 0.00% |
| SPY | ma_50_200 | 9.42% ± 0.00% | 15.40% ± 0.00% | 0.66 ± 0.00 | -32.05% ± 0.00% | 0.0050 ± 0.0000 | 56.20% ± 0.00% |
| SPY | ppo | -5.89% ± 4.73% | 12.35% ± 0.49% | -0.43 ± 0.39 | -45.20% ± 16.35% | 0.4651 ± 0.0188 | 45.26% ± 2.04% |

Buy-and-hold, moving-average crossover, and gradient boosting keep their original
fixed settings. They are repeated identically for all five seed rows to make the
table rectangular; their zero seed std is structural and does not imply certainty
about markets. PPO seed metrics are calculated separately before their mean/std.

## What the checks added

| Asset | Full PPO Sharpe: 0 → 20 bps | 1,024-step PPO Sharpe: ordinary → shuffled | Full PPO random percentile range |
| --- | --- | --- | --- |
| BTC-USD | 0.18 → -0.43 | -0.18 → -0.17 | 46.0–95.2 |
| QQQ | -0.03 → -1.57 | -0.26 → -0.24 | 41.9–62.4 |
| SPY | 0.23 → -1.66 | -0.53 → -0.53 | 20.5–98.0 |

All-in cost includes both fees and slippage. The random percentiles show the range
across the five PPO seeds; a higher percentile does not mean a probability of future
success. Shuffling was tested at a smaller, explicitly matched training budget.
A near-zero Sharpe does not imply zero compounded annual return: gaining 10% and
then losing 10% leaves only 99% of starting wealth. This distinction helps explain
BTC-USD's negative annual growth despite an average PPO Sharpe close to zero.

## Reproduce and inspect

The data end is pinned to `2026-10-30` (exclusive) using the original
cached downloads. `results/final/config.json`, `manifest.json`, and `folds.csv`
record settings, versions, signatures, and boundaries. `metrics.csv` contains
mean/std for all six metrics; `metrics_by_seed.csv` retains every run.
Test decisions ran from 2018-10-19 through
2026-10-28, with the last return realized at
2026-10-29; earlier history supplies training and feature warmup.
Each PPO fit requested 10,000 learning steps;
the library completes its 256-step chunks, collecting
10,240 training steps per fit.
The measured training-and-main-output runtime was 11.1
minutes, below the conservative 21.3-minute pre-run estimate and 30-minute threshold.
`*_oos.csv` gives daily decisions, realized returns, costs, and final liquidation.
Equity curves track compounded account value; drawdown plots track losses from
previous peaks. `*_cost_sensitivity.png` reprices identical positions at 0/5/7/20
bps total cost, including slippage, without retraining policies for each cost.
See `results/checks/report.md` for the paired shuffled-label experiment and its
lower training budget; `results/checks/final_random_policy/` compares the full
agent against random positions calibrated retrospectively to its activity.

## Limitations that matter for this conclusion

- The five seeds reuse the same market history; they are not five independent
  economic experiments or a confidence interval for future profitability.
- Shuffling training returns preserves their mean and can reward market direction;
  it also breaks serial order and the consistency between lagged features and
  simulated reward history. Five permutations per fold are an artificial
  diagnostic, not a formal statistical test or alternate realistic market.
- Cost plots hold positions fixed and do not model how an agent retrained at a
  different cost might behave. Constant slippage can understate difficult execution.
- Random policies match average exposure and turnover, using test positions for
  descriptive calibration; they do not match all PPO behavior or constitute a
  strategy available before the test. Their percentiles are not statistical p-values.
- Adjusted vendor history, surviving asset selection, and common-date alignment
  can carry retrospective information even though code-level timing is causal.
- Rebalancing from price drift, borrowing/funding, liquidity, taxes, execution
  delays, and market impact are omitted. The observation omits some reward history.
- All test periods have now been inspected. Any further change motivated by these
  results requires new untouched data or nested validation, meaning separate
  earlier data used for model choices before a final untouched evaluation.


## Original PPO diagnostic: run_001


**Annualized return** means compounded growth expressed per year; **gross** is
before costs and **net** is after costs. **Turnover** is the absolute change in
target position: 0.5 means changing half an account's target exposure per day.
**Average absolute position** measures exposure size, ignoring direction. **Long**
benefits from rising prices, **short** benefits from falling prices, and **flat**
means a position within 0.00000001 of zero. Values are averages across five seeds
(repeatable random starting points); ± is their sample standard deviation, a
measure of variation. All six original performance metrics remain in `results/final/`.
Entry, direction reversal, fold carry, and final exit costs are included.

| Asset | Before costs, yearly | After costs, yearly | Daily turnover | Average absolute position | Long | Short | Flat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTC-USD | -1.48% ± 10.44% | -10.25% ± 9.42% | 0.528 | 0.636 | 53.7% | 46.3% | 0.0% |
| QQQ | -1.77% ± 1.79% | -9.88% ± 1.47% | 0.488 | 0.597 | 55.1% | 44.9% | 0.0% |
| SPY | 2.17% ± 5.28% | -5.89% ± 4.73% | 0.465 | 0.580 | 51.2% | 48.8% | 0.0% |

## Five-sentence explanation

Before costs, PPO's average yearly compounded returns were BTC-USD -1.48%; QQQ -1.77%; SPY 2.17%.
After costs, they were BTC-USD -10.25%; QQQ -9.88%; SPY -5.89%.
Average daily turnover was BTC-USD 0.528; QQQ 0.488; SPY 0.465, so the agent changed roughly half an account's target exposure each day.
A positive before-cost return alone does not establish a prediction signal, meaning information that reliably helps choose profitable future trades, because market direction and luck also affect returns.
Trading expenses explain a substantial part of the loss, but weak before-cost performance relative to buy-and-hold also matters, so trading less alone is not proven to solve the problem.

![Before and after costs](../results/diagnostics/run_001/before_after_costs.png)
