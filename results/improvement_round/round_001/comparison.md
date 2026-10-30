# Pre-declared improvement round: all outcomes

A **seed** is a repeatable random starting point. **Net return** is growth after
fees and slippage (worse execution prices). **Annualized** means expressed on a
one-year compounded scale. **Volatility** is return variability. **Sharpe** divides
average return by variability and scales to a year. **Drawdown** is a fall from a
previous wealth peak. **Turnover** is the change in target exposure each day; 1.0
means an account's full target exposure changed. **Hit rate** is the fraction of
exposed periods with positive net return. Values following ± are sample standard
deviations: spread across seeds, not confidence about future markets.

## Eight-sentence interpretation

None of the four new variants beat buy-and-hold's average yearly return after costs on any asset.
Of the 12 asset/variant comparisons, 0 exceeded buy-and-hold on both average return and Sharpe, a measure of return relative to variability.
The turnover penalty reduced average daily trading on 3 of 3 assets and improved average return over original PPO on 3 of 3.
The long-only agent improved average return over original PPO on 3 of 3 assets, but buying a historically rising market does not itself show predictive timing.
The volatility-targeted holding rule lowered measured variability on 3 of 3 assets and improved Sharpe on 2 of 3, while earning less, so its benefit is a risk comparison rather than higher absolute return.
The fixed-feature forecast improved average return over the original forecasting rule on 0 of 3 assets, which evaluates the combined fixed model and position rule rather than proving the inputs caused an improvement.
The five seeds show variation from random starting points, while identical rule-based decisions have zero seed spread and neither measures uncertainty across future markets.
Every variant was kept with its declared settings, but these reused test dates make this exploratory evidence that still needs new untouched history before claiming a lasting improvement.

## All strategies, after costs

| Asset | Strategy | Yearly net return | Yearly volatility | Sharpe | Worst drawdown | Daily turnover | Hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTC-USD | Buy-and-hold | 37.16% ± 0.00% | 60.96% ± 0.00% | 0.826 ± 0.000 | -76.64% ± 0.00% | 0.001 ± 0.000 | 49.85% ± 0.00% |
| BTC-USD | Fixed-feature forecast | 7.54% ± 0.00% | 13.73% ± 0.00% | 0.598 ± 0.000 | -20.37% ± 0.00% | 0.120 ± 0.000 | 48.21% ± 0.00% |
| BTC-USD | Original forecast | 24.78% ± 0.00% | 60.96% ± 0.00% | 0.664 ± 0.000 | -77.74% ± 0.00% | 0.376 ± 0.000 | 49.90% ± 0.00% |
| BTC-USD | 50/200 average rule | 32.27% ± 0.00% | 46.23% ± 0.00% | 0.834 ± 0.000 | -60.57% ± 0.00% | 0.005 ± 0.000 | 50.28% ± 0.00% |
| BTC-USD | Original PPO | -10.25% ± 9.42% | 43.52% ± 1.45% | -0.035 ± 0.229 | -84.49% ± 6.92% | 0.528 ± 0.020 | 48.22% ± 1.07% |
| BTC-USD | PPO long-only | 6.76% ± 7.09% | 32.68% ± 2.43% | 0.358 ± 0.222 | -61.17% ± 10.35% | 0.249 ± 0.029 | 49.29% ± 0.64% |
| BTC-USD | PPO + turnover penalty | -9.15% ± 8.79% | 43.74% ± 1.62% | -0.003 ± 0.211 | -84.65% ± 6.78% | 0.502 ± 0.018 | 48.53% ± 0.96% |
| BTC-USD | Volatility-targeted holding | 9.10% ± 0.00% | 12.61% ± 0.00% | 0.754 ± 0.000 | -20.54% ± 0.00% | 0.011 ± 0.000 | 49.85% ± 0.00% |
| QQQ | Buy-and-hold | 20.03% ± 0.00% | 23.47% ± 0.00% | 0.896 ± 0.000 | -36.69% ± 0.00% | 0.001 ± 0.000 | 55.75% ± 0.00% |
| QQQ | Fixed-feature forecast | -2.23% ± 0.00% | 9.01% ± 0.00% | -0.206 ± 0.000 | -23.49% ± 0.00% | 0.147 ± 0.000 | 48.16% ± 0.00% |
| QQQ | Original forecast | 6.40% ± 0.00% | 23.52% ± 0.00% | 0.382 ± 0.000 | -44.51% ± 0.00% | 0.277 ± 0.000 | 52.58% ± 0.00% |
| QQQ | 50/200 average rule | 17.44% ± 0.00% | 20.09% ± 0.00% | 0.901 ± 0.000 | -27.56% ± 0.00% | 0.005 ± 0.000 | 56.69% ± 0.00% |
| QQQ | Original PPO | -9.88% ± 1.47% | 15.95% ± 0.79% | -0.571 ± 0.072 | -61.29% ± 7.47% | 0.488 ± 0.021 | 45.69% ± 1.07% |
| QQQ | PPO long-only | 0.11% ± 0.98% | 11.63% ± 1.34% | 0.069 ± 0.082 | -30.13% ± 5.80% | 0.252 ± 0.017 | 53.02% ± 0.65% |
| QQQ | PPO + turnover penalty | -9.76% ± 1.42% | 16.24% ± 0.85% | -0.550 ± 0.076 | -60.79% ± 7.82% | 0.457 ± 0.017 | 46.37% ± 0.81% |
| QQQ | Volatility-targeted holding | 11.11% ± 0.00% | 10.92% ± 0.00% | 1.020 ± 0.000 | -15.92% ± 0.00% | 0.020 ± 0.000 | 55.70% ± 0.00% |
| SPY | Buy-and-hold | 15.30% ± 0.00% | 18.46% ± 0.00% | 0.864 ± 0.000 | -32.05% ± 0.00% | 0.001 ± 0.000 | 55.85% ± 0.00% |
| SPY | Fixed-feature forecast | -0.75% ± 0.00% | 8.78% ± 0.00% | -0.041 ± 0.000 | -23.99% ± 0.00% | 0.130 ± 0.000 | 47.52% ± 0.00% |
| SPY | Original forecast | 0.23% ± 0.00% | 18.49% ± 0.00% | 0.105 ± 0.000 | -39.18% ± 0.00% | 0.309 ± 0.000 | 51.69% ± 0.00% |
| SPY | 50/200 average rule | 9.42% ± 0.00% | 15.40% ± 0.00% | 0.662 ± 0.000 | -32.05% ± 0.00% | 0.005 ± 0.000 | 56.20% ± 0.00% |
| SPY | Original PPO | -5.89% ± 4.73% | 12.35% ± 0.49% | -0.431 ± 0.392 | -45.20% ± 16.35% | 0.465 ± 0.019 | 45.26% ± 2.04% |
| SPY | PPO long-only | 2.34% ± 1.79% | 8.62% ± 1.31% | 0.326 ± 0.207 | -19.32% ± 7.61% | 0.218 ± 0.015 | 53.34% ± 1.96% |
| SPY | PPO + turnover penalty | -5.49% ± 4.80% | 12.73% ± 0.31% | -0.386 ± 0.396 | -44.54% ± 17.44% | 0.424 ± 0.021 | 45.30% ± 2.30% |
| SPY | Volatility-targeted holding | 9.69% ± 0.00% | 10.43% ± 0.00% | 0.939 ± 0.000 | -13.99% ± 0.00% | 0.020 ± 0.000 | 55.80% ± 0.00% |

The plan was frozen at commit `2e505ef588a26ee56eed190f04e51ec0c87d7218` and implementation at `3e3f1c7aff661f3245bf12197987ca81fff2b99d` before fitting. Original results are reused from `results/final/`; all new paths and failures stay in `results/improvement_round/round_001/`. This round fitted 510 new PPO seed/window policies and evaluated all four variants on five seeds and 17 windows per asset. No settings were chosen from these outcomes.

## Curves and exposure

Curves display the mean of individual account paths; shading shows the full seed range. They are descriptive averages, not an implemented combined strategy. Drawdowns are calculated separately for each seed before display. `exposure_by_seed.csv` retains before-cost returns, trading activity, exposure sizes, and long/short/flat fractions.

## What remains uncertain

The round follows inspection of earlier test results. Pre-declaration prevents additional searches within the round but does not undo that selection. The volatility target only roughly matches risk; long-only gains can reflect market direction. The forecast changed its loss, tree settings, and position scaling together. Vendor adjustments, survivor selection, shared calendars, price-drift rebalancing, funding, short borrowing, and simplified fills remain limitations. No future-profit or formal significance claim is made.
