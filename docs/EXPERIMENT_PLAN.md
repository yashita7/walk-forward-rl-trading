# Frozen improvement plan: round_001

This plan is written before fitting any new variant. A **variant** is one fixed
change to the original experiment. **Hyperparameters** are settings chosen before
learning, such as training steps or tree depth. Their exact values are in
`config.yaml`, which must be committed to Git before the round starts. Git records
versioned file snapshots; the runner records the commit identifier and rejects
uncommitted changes to the plan, configuration, or training code.

## What is fixed

Use SPY, QQQ, and BTC-USD, original cached prices from 2015-01-01 through the
exclusive end date 2026-10-30, and the same common observed dates. **Walk-forward**
means learning on each earlier three-year window and testing on the following
up-to-six months. Keep all 17 test windows per asset and seeds 7, 21, 42, 84, 123;
a **seed** fixes a random starting point. Purge training answers ending at or
after the next test boundary. **Purging** means excluding answers known too late.
Inputs use only completed bars before the opening trade; positions earn the next
open-to-open return. All scalers and models use training rows only.

Actual fees remain 5 basis points plus 2 basis points slippage per unit of position
changed. A **basis point** is 0.01%; **slippage** means a worse fill than the stated
price. Carry positions between windows and charge first entry and final exit.
Use 252 periods per year, as in the original aligned equity-session calendar.
All variants use the identical test dates, with no test-based early stopping or
parameter selection. Archive original results; never overwrite them.

## Exactly four variants

| Variant | Fixed change | Question |
| --- | --- | --- |
| `ppo_turnover_penalty` | Original PPO (Proximal Policy Optimization, learning actions from rewards), with reward additionally reduced by `0.001 * abs(new_position - previous_position)`; action range stays [-1, 1]. | Does explicitly discouraging position changes reduce trading expenses and improve later results? |
| `ppo_long_only` | Original reward; action range [0, 1], with no extra turnover penalty. A long position gains when price rises; zero holds cash. | Does preventing bets on falling prices help on these historically rising assets? |
| `vol_target_buy_hold` | Hold `min(1, 0.10 / max(recent_annual_volatility, 0.01))`; volatility is the sample standard deviation of the previous 20 close-to-close percentage returns, times sqrt(252), delayed one session. | Does a simple rule with reduced exposure compare more fairly with PPO's variable risk? |
| `gb_fixed_features` | Gradient boosting (adding small prediction trees) with Huber loss, 200 trees, learning rate 0.03, depth 2, minimum 20 observations per leaf, and exactly five inputs listed below; exposure is scaled prediction, clipped to [-1, 1]. | Can a more regularized forecast with a small fixed input set compete with the original forecast and PPO? |

**Volatility** means variability of returns. A 10% target is a sizing rule, not a
guarantee of future risk, and no borrowing beyond account value is allowed.
The volatility rule calculates risk before the trade, never from future returns.
It can still trade daily and pays the same costs. It is a rough risk comparison,
not exact matching of PPO's changing exposure or risk.

The forecast inputs are `log_return`, `return_lag_1`, `volatility_20`,
`momentum_20`, and `ma_spread`, all already delayed by `decision_frame`.
A **log return** is log(new price / old price); a **lag** is an older value;
**momentum** is the recent 20-session price change; **MA spread** compares the
50-session and 200-session price averages. **Huber loss** reduces the influence
of unusually large prediction errors. A **leaf** is a final branch of a tree;
requiring 20 training examples limits fragile tiny groups. The scaled position
is `prediction / max(training_target_std, 0.000001)`, clipped to [-1, 1];
the scale factor is fixed at 1.0 and its standard deviation uses training labels
only. Here "stronger" names a proposed candidate, not an established improvement.
The forecast is fitted once for each declared seed per fold; no feature selection,
search, or validation-based early stopping is performed.

The two PPO variants each request 10,000 training steps per asset/fold/seed,
collecting 10,240 because the library finishes 256-step batches. Each uses 64-row
training batches, ten passes through each collected batch, learning rate 0.0003,
discount factor 0.99 (how future rewards are weighted), advantage-smoothing factor
0.95 (how noisy reward estimates are averaged), update clip 0.2 (limits changes),
no entropy bonus (no extra reward for random actions), value-loss weight 0.5,
gradient clipping at 0.5, and two 32-unit hidden layers with Tanh activation.
A **hidden layer** is a trainable intermediate calculation; **Tanh** maps numbers
between -1 and 1. Adam is the fixed numerical optimizer, with epsilon 0.00001;
these settings match the original relevant PPO defaults. All numerical settings,
including the unchanged recent-return volatility penalty 0.01, are explicit in
`config.yaml`. Runtime library versions are saved before the round.

## Reporting and decision rule

Run each of these four variants once under the frozen setup, keeping all five
seed paths. Reuse, rather than refit, the original buy-and-hold, 50/200 crossover,
gradient-boosting, and PPO paths from `results/final/` as references.
Report all eight strategies: mean and sample standard deviation (spread across
seeds) of yearly compounded net return, yearly volatility, Sharpe, worst drawdown,
daily turnover, and hit rate. **Sharpe** is average return divided by variability,
scaled to a year; **drawdown** is a fall from a prior wealth peak; **hit rate** is
the fraction of exposed periods with positive returns after costs.
Deterministic rules are repeated for the same five seed rows; zero seed spread is
structural, not evidence of certainty about future markets.

Compare every variant with buy-and-hold on both mean yearly net return and mean
Sharpe, and report both measures even when they disagree. Lower losses or higher
Sharpe do not automatically mean higher absolute return. Do not select a winner
by changing the metric, choosing a favorable asset/seed, or dropping a failed run.
Save individual seed metrics and exposure diagnostics, equity curves (compounded
account value), and drawdowns. Record errors and incomplete runs as well as successes.
Resume an interrupted identical run from checkpoints (saved predictions), without
changing its settings; any correction must be recorded and preserve earlier files.

## Honesty boundary

The original test results and the cost diagnostic were seen before this plan.
This round is **exploratory**, meaning a follow-up investigation on already
examined history. Pre-declaring these four user-requested ideas prevents further
search within this round; it cannot turn reused dates into an untouched final
test. Report the negative or mixed outcome directly. New untouched dates or a
separate earlier validation design are still needed before claiming improvement
that will generalize, meaning work beyond this inspected history.
