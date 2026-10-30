# Start here

This project checks whether rules learned from past daily prices can choose useful
trading positions. A **position** is the fraction of account value exposed to a
price change: +1 gains when prices rise, 0 holds cash, and −1 gains when prices fall.
SPY and QQQ are traded funds; BTC-USD is Bitcoin priced in US dollars.

We compare **baselines**, simple reference rules, with **PPO** (Proximal Policy
Optimization), an algorithm that learns actions from numerical feedback called a
**reward**. **Buy-and-hold** means buying once and staying invested.

The completed five-run study found that PPO lost to buy-and-hold on all three
assets after trading expenses. Average yearly compounded returns were −5.89% for
SPY, −9.88% for QQQ, and −10.25% for BTC-USD; buy-and-hold returned +15.30%,
+20.03%, and +37.16%, respectively. **Compounding** means each return acts on the
account's updated value. Five **seeds**, repeatable random starting points,
revealed variation between trained agents; they are not five independent markets.

**Walk-forward evaluation** means repeatedly learning on earlier dates and testing
on the following dates. **Look-ahead bias** means using information before it
would have been available. We delay inputs until the next opening session, keep
training separate from testing, and remove training answers that become known too
late. Those controls do not prevent every possible data problem.

A **shuffled-label check** scrambles training answers while leaving test returns
unchanged. At a smaller, matched training budget, this barely changed PPO's
results. That gives little evidence of useful return timing at that budget;
nonzero results can still come from market direction, luck, or costs.

A **random-agent check** compares PPO with 1,000 programs making random long/short
decisions per asset, with similar average exposure and trading activity. It shows
how widely luck can vary on this history, not a probability of future success.

Next we separate before-cost performance from expenses, then freeze four proposed
changes before testing them: penalize trading, allow buying only, adjust a holding
rule's exposure to recent price variability, and strengthen the forecasting rule.
All successes and failures will remain recorded. These dates have already been
examined, so the next round is exploratory: new untouched dates are still needed
before claiming that any improvement generalizes.
