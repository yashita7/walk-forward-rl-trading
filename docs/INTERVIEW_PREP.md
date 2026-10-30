# Fifteen interview questions, with honest short answers

Use these to practice explaining your own implementation. The follow-up round is
exploratory: it reuses dates whose earlier results were already inspected.

1. **What did you build?** A Python experiment that compares trading rules using
   past daily prices and tests them on later dates, after trading expenses. The
   result is a careful negative or mixed finding, not a profitable trading product.

2. **What are features and labels here?** Features are inputs: recent price changes,
   variability, and averages. The label, or training answer, is the next opening
   price divided by today's opening price, minus one (`src/features.py:28`).

3. **What is look-ahead bias, and how do you avoid it?** It means using information
   before it was available. Features look backward, observations are delayed,
   fitted transforms use training dates only, and tests mutate future data to
   check that earlier inputs stay unchanged.

4. **Why is there a negative shift in the code?** `shift(-1)` builds tomorrow's
   return as the answer, not an observation the agent can see. Predictors and
   targets are separate objects; passing targets into observations would be a bug.

5. **Why next-open execution?** Execution means the assumed time and price of a
   trade: the model uses completed earlier bars and trades at the next opening
   session. This avoids assuming an already-known closing price was still
   available, though real delays and crypto's continuous trading remain limitations.

6. **What is purging?** Purging removes training examples whose answers finish at
   or after the next test boundary (`src/backtest.py:36`). A row with an early date
   can still contain an answer that would only become known later.

7. **How do you keep scaling honest?** A scaler subtracts a feature's average and
   divides by its variability. It is fitted only to training rows; test rows use
   that fixed transform (`src/train_rl.py:19`).

8. **Why five seeds?** A seed fixes a random starting state, including network
   weights and training sampling. The mean and standard deviation, or average and
   spread, show variation between runs on this same history—not uncertainty across
   independent future markets.

9. **What does the environment give the agent?** A Gymnasium environment is the
   observation/action/reward interface: here it gives 20 rows of seven features
   plus current position, totaling 141 numbers. Position is exposure, with −1 a
   bet on falling prices, 0 cash, and +1 full buying exposure; the buying-only
   variant restricts this to [0, 1].

10. **How is reward different from money earned?** Reward is the numerical teaching
    score: position times next return, minus real trading costs, minus a penalty
    for recent variability and, in one variant, extra position changes. Only real
    costs are deducted in reported investment returns; teaching penalties are
    not charged again as money (`src/env.py:79`).

11. **Why did original PPO lose?** PPO, Proximal Policy Optimization, learns actions
    by adjusting a neural network, a stack of trainable calculations, from rewards.
    Its average yearly return before expenses was only +2.17% on SPY and negative
    on QQQ/BTC, and changing roughly half an account's target exposure daily made
    costs substantial; these observations do not identify one proven cause of failure.

12. **What does the shuffled-label test establish?** It scrambles training answers
    but keeps real test returns; at the matched 1,024-step budget, PPO changed very
    little. This provides little evidence of useful return timing at that budget,
    but zero profit is not guaranteed because market direction, luck, and expenses
    remain, and shuffling creates an artificial training task.

13. **Why compare random agents?** A random agent chooses exposure without trying
    to predict prices; 1,000 such runs per asset show how much outcomes can vary
    through luck. Matching their average exposure and turnover, meaning position
    changes, uses already observed test positions, so percentile ranks are
    descriptive comparisons rather than formal proof of skill.

14. **What happened in the improvement round?** Four variants were committed before
    fitting: a trading penalty, buying-only PPO, volatility-targeted holding
    (less exposure when recent returns vary more), and a fixed-feature forecast.
    None beat buy-and-hold's average yearly return, although the volatility rule
    improved Sharpe—return relative to variability—on SPY and QQQ; all negative
    outcomes remain saved.

15. **What would you try next, and what can still go wrong?** I would freeze choices
    and collect new untouched dates before making further comparisons, then
    improve execution assumptions and test another data source. Revised prices,
    selecting assets that survived until today, reused test dates, missing-data
    calendars, financing, and market impact—orders moving prices—remain risks;
    more changes on the same dates would not create independent evidence.

If pressed on a number, open `results/improvement_round/round_001/metrics.csv`,
`results/diagnostics/run_001/metrics.csv`, or `results/checks/report.md`.
Say "I have not tested that" when asked about live execution, actual capital,
capacity, or future profitability; none is implemented or established.
