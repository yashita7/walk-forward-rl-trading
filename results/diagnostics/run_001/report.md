# Why did original PPO lose?

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
