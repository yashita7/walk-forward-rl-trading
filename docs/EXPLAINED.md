# rl-trader, explained for a first interview

## 1. The big picture in five sentences

This project asks whether rules learned from past daily prices can choose useful trading positions, meaning how much money to expose to price changes.
It compares simple rules with a price-change predictor and reinforcement learning, meaning learning actions from numerical feedback.
Walk-forward evaluation means repeatedly learning on an earlier time window and testing on the following window, which imitates the order in which information becomes available.
Trading costs are deducted so that frequent trading cannot look profitable just because changing positions is assumed to be free.
The results answer how these particular rules performed on these historical test windows under the stated assumptions, rather than whether they will make money in the future.

## 2. Glossary

| Term | Plain-English meaning and this project's example |
| --- | --- |
| Log return | `log(new_price / old_price)`, using the natural logarithm; `log_return` describes the change between two closes, and adds cleanly across time. |
| Volatility | How much returns vary; `volatility_20` is the sample standard deviation, a measure of spread, of the last 20 log returns. |
| Sharpe ratio | Average return divided by return variability, scaled to a year; `metrics` uses returns after costs and assumes cash earns zero interest. |
| Max drawdown | Worst fall from a previous wealth peak; wealth falling from 100 to 80 gives a drawdown of -20%. |
| Turnover | Amount of target exposure changed; going from -1 to +1 has turnover 2. |
| Transaction cost and slippage | A fee for trading and the gap between an assumed price and a worse actual fill; defaults are 5 and 2 basis points respectively, where one basis point (bp) is 0.01%. |
| Look-ahead bias | Accidentally using information that was unavailable when deciding; using tomorrow's price as today's input would do this. |
| Purging | Removing training examples whose answers become known too late; `walk_forward` excludes outcomes ending at or after the next test boundary. |
| Walk-forward validation | Repeatedly train on earlier dates and test on later dates; this repo uses three-year training windows and six-month test windows. |
| PPO | Proximal Policy Optimization, a reinforcement-learning algorithm that gradually adjusts a neural network (trainable weighted numerical calculations) using rewards; `train_predict` fits it on past prices. |
| Gymnasium environment | A standard interface that gives an agent (the action-choosing program) observations and rewards after actions; `TradingEnv` simulates exposure and price changes. |
| Reward function | The score used to teach an agent; here it is trading return after costs minus a small penalty for recent return variability. |
| Baseline | A simple comparison rule; buying and holding SPY tests whether a complicated agent adds anything over staying invested. |

Some additional words used below: an **asset** is something traded, such as SPY;
a **bar** is one period's open, high, low, close, and volume (number of units traded);
**adjusted prices** are historical prices modified for corporate actions such as
share splits and distributions. A **feature** is a numerical input, such as recent
volatility. A **label** or **target** is the answer a predictor learns to estimate,
here the next open-to-open return. A **model** is a fitted rule that turns inputs
into predictions. **Gradient boosting** combines small decision trees, simple
branching prediction rules, to improve their joint forecast. A **policy** maps an
observation to an action. **Out-of-sample (OOS)** means dates outside the window
used to fit that particular model. A **fold** is one training/test pair. A
**seed** fixes a random-number generator's starting state so a run can be repeated.
The PPO policy is a **neural network**, a stack of trainable weighted calculations
that converts inputs to an action. A **continuous** action can be any number in
its permitted range, rather than a choice from a short list.
**Net** means after costs, while **gross** means before costs. **Annualized** means
expressed on a one-year scale, not a promise about next year.

## 3. Tour of the source files

| File | Most important function/class | What it does and why it exists |
| --- | --- | --- |
| `src/__init__.py` | Package marker | Makes `src` importable so `python -m src.run_all` works. |
| `src/data.py` | `download_bars` | Downloads and caches adjusted daily bars; `clean_bars` removes bad rows and `align_bars` keeps common dates without inventing prices. This isolates data handling from learning. |
| `src/features.py` | `decision_frame` | Builds backward-looking inputs with `make_features`, delays them for next-open trading, and returns separate features, future returns, and outcome dates. This makes information timing explicit. |
| `src/baselines.py` | `baseline_positions` | Produces buy-and-hold, 50/200 moving-average (MA, average of recent prices) crossover (long when the 50-day average exceeds the 200-day average), and gradient-boosting positions. These are the comparisons the agent must justify beating. |
| `src/env.py` | `TradingEnv.step` | Converts an action to exposure, earns the next return, deducts costs, and computes a learning reward. `net_step` is shared with the backtester to keep accounting consistent. |
| `src/backtest.py` | `walk_forward` | Constructs purged chronological folds. `account` applies positions to real returns, and `metrics` grades only the joined OOS history. This is the evaluation foundation. |
| `src/train_rl.py` | `train_predict` | Fits a fresh PPO policy and scaler, a transform that subtracts each training feature's mean and divides by its standard deviation. It then makes sequential test decisions without further fitting. |
| `src/run_all.py` | `run` | Connects the pieces, carries positions across folds, and writes metrics, audit files, and plots. It provides the normal command-line entry point. |

The later `src/checks/` package contains independent diagnostic scripts: see its
`README`-style module docstrings and `results/checks/report.md`. A **diagnostic** is
a deliberate stress test of the experiment, not another attempt to improve test returns.

| Check file | Most important function | Why it exists |
| --- | --- | --- |
| `checks/__init__.py` | Package marker | Keeps diagnostic commands in their own importable package. |
| `checks/common.py` | `evaluate` | Runs the same purged folds for ordinary or shuffled training and saves repeatable experiment details. |
| `checks/run_checks.py` | `main` | Freezes a check plan, runs ordinary/shuffled comparisons, and produces the check report. |
| `checks/cost_sensitivity.py` | `reprice` | Applies new fees to the same decisions, isolating the cost effect. |
| `checks/random_policy.py` | `random_positions` | Creates random long/short decisions to compare PPO against luck at similar activity. |
| `checks/estimate_runtime.py` | `main` | Measures two full-budget fits and estimates the full job before it starts. |
| `checks/run_final.py` | `validate_runtime` | Checks the runtime estimate and runs the five-seed final experiment without silently exceeding the user's time threshold. |
| `checks/report.py` | `final_report` | Converts actual saved numbers into the results explanation and README table. |

The forecasting model learns to predict tomorrow's return directly. PPO instead
learns which action receives better rewards, including the expense of changing
from its previous position. A log return is an input here, while account profit
uses the ordinary percentage return: a price move from 100 to 101 has log return
about 0.00995 but ordinary return exactly 0.01. Mixing those formulas would make
accounting harder to interpret, so the code keeps them separate.

## 4. Walk through one trading day

Line references below point to the actual source, counted from one. Start with the
existing implementation before reading the diagnostic scripts.

1. **Read a completed bar.** `download_bars` (`src/data.py:22`) returns `bars`, a
   pandas DataFrame, meaning a dated table. Its `open` and `close` columns hold
   opening and closing prices. `clean_bars` (`src/data.py:11`) removes unusable rows.
   `align_bars` (`src/data.py:41`) keeps dates observed for every configured asset;
   it never fills a missing price with a later price.
2. **Turn history into features.** `make_features` (`src/features.py:8`) uses
   `close` at line 10. `log_return` at line 11 is a change in logged closing prices.
   The trailing 20-row standard deviation appears at line 21. **Momentum**, the
   change over a longer interval, is the 20-session price change at line 22.
   **RSI**, Relative Strength Index, compares recent gains with recent losses at
   lines 12–16; this code scales it to 0–1 at line 23. None uses a future bar.
3. **Put features on an executable decision date.** At `src/features.py:33`,
   `x = make_features(bars).shift(1)` puts yesterday's completed inputs on today's
   row. At line 34, `returns` contains tomorrow's open divided by today's open,
   minus one. The negative shift here creates the answer, not an input. `ends`
   at line 35 records when that answer becomes known. A missing answer or warmup
   feature causes that example to be removed at lines 36–37.
4. **Make an observation.** In training, `TradingEnv._observation`
   (`src/env.py:47`) flattens the last `self.window` rows of `self.features` into
   one vector and appends `self.position`. With seven features and a 20-session
   window, that is 141 numbers. Test inference builds the same shape in
   `train_predict` at `src/train_rl.py:37`; the scaler was fitted only on
   `train_x` at `src/train_rl.py:19`. **Inference** means
   using an already fitted model to choose an output.
5. **Choose an action and position.** `model.predict(..., deterministic=True)`
   at `src/train_rl.py:38` uses the policy's chosen action without sampling extra
   randomness. `position` at `src/env.py:64` clips an action to [-1, 1]. A value
   of +0.5 means half the account exposed to rising prices; 0 means cash; -0.5
   means a short exposure, which benefits from falling prices and loses if they rise.
6. **Earn the next return and compute the cost.** Suppose `self.position` was
   0.2, the new `position` is 0.5, and `self.returns[self.i]` is +0.01 (1%).
   `net_step` (`src/env.py:11`) calculates `turnover = abs(0.5 - 0.2) = 0.3`
   at line 14. With `cost_rate = 0.0007`, `cost` at line 15 is 0.00021, or
   2.1 bps of account value. Gross return is `0.5 * 0.01 = 0.005`; net return
   at line 16 is `0.00479`, or 0.479%. Cost is deducted *inside* this calculation,
   rather than after the reward or a second time later.
7. **Give the agent feedback.** `TradingEnv.step` calls `net_step` at
   `src/env.py:67`, stores `net` in `self.history` at line 68, calculates `risk`
   at line 69, and returns `net - self.penalty * risk` at line 72. If `risk`
   were 0.004, the default penalty of 0.01 would subtract another 0.00004 from
   the learning reward. This penalty is a teaching preference, not money paid.
   `info['net_return']` at line 73 retains the actual net return for comparison.
8. **Grade the real test trade.** `account` (`src/backtest.py:45`) uses the same
   `net_step` at line 59. At the end of the entire test history, lines 65–73
   charge a final exit. `metrics` at line 84 grades the real joined test returns,
   never the risk-penalized teaching reward. Positions persist between folds.

Training time and trading time differ: the simulator already has historical
answers so it can teach the agent *after* each action. The observation never
contains those answers. In actual test inference, the agent only sees inputs and
its previous position; accounting applies the realized answer afterward.

## 5. Questions an interviewer will ask

1. **What question does your experiment answer?** Whether these fixed rules, on
   these assets and chronological test windows, earned useful returns after our
   modeled costs. It does not prove future profitability.
2. **How do you prevent look-ahead bias?** Backward-looking features, a one-session
   input delay, training-only fitting, purged labels, and future-data perturbation
   tests. Vendor revisions and choices made after viewing results remain risks.
3. **Why next-open execution?** The day's final close is known only when the bar
   completes. Using it while assuming an already achieved close fill is optimistic;
   the next opening session gives a clearly later execution point.
4. **What does purging remove?** A training label that ends at or after the test
   boundary. Being attached to an earlier date does not mean its answer was known
   early enough to train on.
5. **Why multiple seeds?** Neural-network starting weights and training sampling
   change results. Mean and standard deviation, meaning average and spread across
   runs, reveal this variation. They are not independent market experiments.
6. **Why might PPO fail here?** The useful price signal may be very weak, costs
   punish unnecessary actions, history is limited, and relationships can change.
   PPO solves its training task, which need not match future market conditions.
7. **Why use the simple comparisons?** Rising markets can reward anyone staying
   long. Buy-and-hold and the MA rule show whether extra complexity helped; the
   forecasting baseline separates prediction from reinforcement learning.
8. **Are your trading costs realistic?** They are explicit and nonzero, including
   position reversals and the final exit. They still simplify changing spreads,
   financing, short availability, rebalancing, and market impact (our orders moving prices).
9. **Is a positive shuffled-label result automatically a bug?** No. Shuffling
   training answers destroys their alignment with features, but preserves their
   average; market direction, random luck, and costs still affect real test returns.
   Persistent unusual performance deserves investigation, not automatic rejection
   or automatic acceptance.
10. **What would you do next?** Freeze the research choices and evaluate on new
    untouched dates, add more realistic costs, and test whether results survive
    independent datasets. I would not keep changing settings to win on these tests.

## Read the result honestly

An **equity curve** tracks account value after compounding returns, meaning each
new return acts on the previous account value. A **regime** is a period with
different market behavior. A good-looking curve can reflect a favorable regime,
exposure to a rising asset, or luck. A random-policy distribution, the collection
of outcomes from many randomly acting agents, helps show how wide luck can be.
It is a comparison, not proof of statistical significance (strong evidence beyond
what randomness alone could reasonably explain).

## What does a 10,000-step training budget mean?

A **learning step** is one simulated action followed by the historical return and
reward. An **episode** is one pass through the permitted training transitions;
after it ends, the agent starts again from cash. Ten thousand steps therefore
revisit the same roughly three years of training dates, rather than supplying
ten thousand new observations. PPO gathers experience in chunks of 256 steps in
this configuration; the library completes the final chunk, so requesting 10,000
actually collects 10,240 steps per fitted policy. These extra steps stay in the
training window. Every fold starts a fresh policy; test trading carries positions
between folds, but never updates a policy from an upcoming test return.
