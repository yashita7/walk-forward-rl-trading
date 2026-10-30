# Explain it out loud

Read slowly and pause after the result. Timing assumes roughly 130–140 spoken
words per minute; use a timer and adjust your pace, not the claims.

## One minute

I built a Python project that checks trading ideas using past prices.
A position means how much of an account gains or loses when a price changes.
I compared simple buying rules with a predictor and PPO, an algorithm that learns
actions from numerical rewards.

I trained on earlier dates and tested on later dates, repeatedly moving forward
through time.
Inputs used only information available before each trade, and I deducted trading
expenses.

The original agent lost to buy-and-hold on all three assets.
I then committed four proposed changes before running them and kept every result.
None beat buy-and-hold's average return after costs.

My main lesson was that honest testing matters more than a complicated model.
The follow-up dates were already inspected, so new untouched history is still
needed before claiming a lasting improvement.

## Three minutes

My project asks whether a learned trading rule adds value beyond simple buying
rules after trading expenses.
It uses daily prices for SPY and QQQ, two traded funds, and Bitcoin priced in
US dollars.
A position is the fraction of account value exposed to the next price change.
Positive positions benefit when prices rise; negative positions benefit when
they fall.

I compared buy-and-hold, which stays invested, a rule based on recent price
averages, a forecasting model, and Proximal Policy Optimization, or PPO.
PPO learns actions from a reward, a numerical teaching score.
Its inputs describe recent price changes and variability, plus its current
position.

The evaluation uses walk-forward testing: learn from an earlier three-year window,
then test on the following months, and repeat.
There are seventeen test windows per asset and five random starting points for
each learned strategy.
Those starting points reveal training variation; they are not independent markets.

I focused on look-ahead bias, meaning information being used before it was known.
Features use completed earlier prices, and a position is chosen for the next
opening trade.
Scalers, which put inputs on comparable numerical scales, are fitted on training
dates only.
Purging removes training answers that would become known too late.
I also charge first entries, changes of direction, and the final exit.

The original PPO agent underperformed every comparison rule on average after
costs.
Before costs, its yearly return was weak on SPY and negative on QQQ and Bitcoin.
It also changed roughly half an account's target exposure each day, making fees
important.

I checked scrambled training answers and compared outcomes with one thousand
random agents per asset.
These checks help show whether apparent success could reflect market direction
or luck; they do not prove that a strategy will work later.

I then committed four changes before running them: penalize trading, allow buying
only, reduce a holding rule's exposure when prices vary more, and use a fixed
forecasting candidate.
All used the same dates and costs.
None beat buy-and-hold's average yearly return.
The risk-controlled holding rule had a better return-to-variability ratio on SPY
and QQQ, while earning less.

I kept the failed candidates and did not adjust their settings to improve these
numbers.
Because the follow-up reused inspected history, it remains exploratory.
New untouched dates and more realistic trading assumptions are necessary before
making stronger claims.
