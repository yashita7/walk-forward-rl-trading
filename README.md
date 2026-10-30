# rl-trader

A small Python 3.11 research pipeline combining causal time-series features,
gradient boosting, and PPO with rolling walk-forward evaluation. The goal is an
auditable experiment, not a claim of profitable trading. No live trading or notebooks.

Start with [the beginner interview guide](docs/EXPLAINED.md). The completed
five-seed, full-budget study is explained in [the results summary](docs/RESULTS_SUMMARY.md),
with its mean/std table at `results/final/metrics.csv` and checks at
[`results/checks/report.md`](results/checks/report.md). The current suite has
**20 passing tests**; the older smoke results below remain an integration example.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.run_all
```

Run from this directory. All assets, dates, costs, window sizes, seeds, and training
budgets live in `config.yaml`. `end: null` resolves to today's UTC date, exclusive,
so the current daily bar is excluded. The first download requires network access.
Adjusted OHLCV is cached in `data/*.parquet` using the symbol, date range, and
adjustment policy as the key. An unchanged request reuses its cache; remove the
specific cache file to refresh. Invalid or missing prices are dropped, never filled.

For a quick real-data integration check:

```bash
python -m src.run_all --smoke
```

This uses the first two test folds, 256 PPO timesteps per fold, and **all assets and
configured seeds**. This tiny training budget checks execution; it cannot assess PPO's
potential. No parameters are chosen by these test results. For an explicit offline
fixture, run `python -m src.run_all --smoke --synthetic`. Synthetic outputs are saved
separately and are never substituted after a failed market download.

## Timing and leakage controls

`features.make_features` stamps each feature at its completed bar: log returns,
two return lags, trailing 20-session volatility and momentum, 14-session RSI, and
the 50/200 MA spread. All windows are backward-looking. Warmup NaNs are dropped.

The executable decision table delays these features by **one full session**:

```
completed close t-1 -> observation -> action at open t -> return at open t+1
```

Thus a position chosen at decision t earns exactly the next-period return. Using
the known close t to assume a fill at that same close would be optimistic; this
project instead uses next-open execution with costs. Current open/high/low/close
and the next-return target are never part of the observation. Target and outcome
timestamps are kept separately. RSI uses simple trailing average gains/losses;
50/200 crossover is long when the fast MA exceeds the slow MA, otherwise cash.
Gradient boosting fits the next open-to-open return and trades its sign, long or
short, with fixed hyperparameters. Buy-and-hold is continuously long.

Each asset gets a separate forecast and PPO policy. Walk-forward windows use three
calendar years of usable feature history, followed by six calendar months of test
decisions; the boundary advances six months. The initial 200-bar warmup precedes
the first training window. The last test block can be partial. Training labels must
end **strictly before** the test boundary, purging the last overlapping transition.
`folds.csv` records the actual bounds and latest training outcome for inspection.

Scalers and forecasting models are fitted fresh using only each fold's training
rows. PPO uses a fresh CPU policy per asset/fold/seed and train-only standardization.
Training observations stay inside the training window. A separate, unfitted next
feature row supplies the final next-state observation for Gymnasium timeout
bootstrapping; that row is also before the test boundary. Test observation windows
can use already observed historical context. PPO test actions are deterministic;
there are no updates, test-based early stopping, or test-based model selection.

The action space is continuous `[-1, 1]`; the observation is the flattened recent
feature window plus current exposure. Training reward is:

```
net_return = position * next_open_return
             - (cost_bps + slippage_bps) / 10000 * abs(position - previous_position)
reward = net_return - volatility_penalty * trailing_std(net_returns)
```

The risk window includes the current realized return and uses population standard
deviation. It affects learning rewards only; reported returns retain actual net
P&L. Training episodes start from cash. Finite training episodes truncate, with a
valid next observation; they do not force a liquidation in the learning reward.
The API follows [Gymnasium's environment contract](https://gymnasium.farama.org/main/api/env/).

OOS paths start in cash, charge their first entry, carry exposure across retraining
boundaries, and charge one final liquidation at the final outcome open. There is
no free reset at a fold boundary. The recorded last position is the exposure held
during that final interval; `liquidation_turnover` records the subsequent exit.
Moving from -1 to +1 costs 14 bps under the default 5 bps fee + 2 bps slippage per
unit of one-way turnover. All costs, including the final exit, use beginning-of-
interval NAV units under the stated additive reward convention.

## Results

Outputs are under `results/market/`, `results/market_smoke/`, or
`results/synthetic_smoke/`. Each run in the same mode overwrites its matching files.

| File | Contents |
| --- | --- |
| `metrics.csv` | One row per asset, strategy, and PPO seed; all on concatenated OOS returns |
| `metrics_summary.csv` | Per-asset/strategy mean and sample std across seeds; baselines have no seed std |
| `*_oos.csv` | Decision time, outcome time, asset return, held position, turnover, costs, net return, and fold |
| `folds.csv` | Training/test boundaries and last permissible training outcome |
| `manifest.json` | Effective settings, exact runtime versions, data ranges, mode, and generation time |
| `*_equity.png`, `*_drawdown.png`, `*_positions.png` | Equity/drawdown for all strategies and seeds; PPO positions |

The results table columns have these definitions:

| Metric | Definition |
| --- | --- |
| Annualized return | Geometric compounded net return raised to `annualization / observations` |
| Annualized volatility | Sample daily standard deviation times square root of annualization |
| Sharpe | Mean net return / sample std times square root of annualization; zero risk-free rate |
| Max drawdown | Most negative equity / running peak - 1; includes starting NAV 1 |
| Turnover | Mean absolute target exposure change per observation, including entry and final exit |
| Hit rate | Fraction of exposed periods with positive net return; flat periods excluded |

Verified real-data smoke run: 251 OOS decisions per strategy, from 2018-10-19
through 2019-10-18, with the final outcome at the next observed open. The three PPO
seeds produced the following **mean ± sample std**. These are 256-step integration
checks, not tuned or full-budget research results; the full 2015-to-present data
was downloaded, but this run intentionally evaluated only the first two folds.

| Asset | PPO annualized net return | PPO Sharpe | PPO max drawdown |
| --- | --- | --- | --- |
| SPY | -2.30% ± 0.98% | -2.08 ± 0.47 | -2.98% ± 1.29% |
| QQQ | -2.69% ± 3.06% | -1.10 ± 1.83 | -3.92% ± 3.50% |
| BTC-USD | 1.10% ± 6.52% | -0.08 ± 1.28 | -8.41% ± 2.36% |

The full baseline comparison and all six metrics are in
[`results/market_smoke/metrics.csv`](results/market_smoke/metrics.csv); seed
summaries are in [`metrics_summary.csv`](results/market_smoke/metrics_summary.csv).
Verification completed with **14 passing tests** and nine generated plots.

Zero-volatility Sharpe is reported as zero by convention. Bankruptcy or nonfinite
returns fail explicitly. PPO mean/std are computed from each seed's full-path
metrics, **not** metrics of a return path averaged across seeds. Baselines use seed
`-1` as a CSV sentinel. No pooled multi-asset portfolio is implied.

The default assets are SPY, QQQ, and BTC-USD, aligned to their common observed
dates with no price filling. This samples BTC on the equity session calendar and
makes crypto returns span omitted weekends. Annualization defaults to 252; set it
to 365 when using BTC alone on its daily calendar. BTC and equities have different
session clocks, so same-date rows are not synchronized intraday cross-asset data.
Each policy only uses its own asset's history; no cross-asset features are used.

## Tests

`test_no_lookahead.py` mutates future prices, compares full/prefix feature output,
checks exact next-open timing, checks purged and nonoverlapping folds, and verifies
that alignment never fills missing dates. It also audits actual scaler fitting and
perturbs future test features/targets to verify the first PPO action is invariant.
`test_costs.py` checks entry, reversal,
fold carry, final liquidation, environment/accounting parity, risk reward separation,
and the Gymnasium API. `test_metrics.py` checks hand-computed geometric returns,
sample volatility, Sharpe, initial-loss drawdown, turnover, and hit rate.

## Limitations

- **What the new checks establish:** shuffled training labels retain their average,
  so real OOS returns need not equal zero. Five permutations per fold are a
  diagnostic, not a formal significance test. Cost sensitivity reprices fixed
  decisions and does not retrain an agent for each fee. Random-policy comparisons
  match test-period exposure and turnover retrospectively; their percentile ranks
  describe this history and are not probabilities of future skill.
  Shuffling also breaks return order and the match between lagged real-return
  features and simulated rewards, so it is an artificial control rather than
  another realistic price history.
- **Seed interpretation:** all seeds see the same market history. Deterministic
  baselines are repeated for a consistent table and have zero seed spread by
  construction. Neither their zero spread nor PPO's five-seed variation estimates
  uncertainty across future market histories. The final run uses five seeds and
  the original full training budget; diagnostics use an explicitly smaller budget.

- **Overfitting and selection:** choosing assets, features, or hyperparameters after
  examining these results is test-set tuning, even if no code fits test rows.
  Reserve a new untouched holdout or use nested validation before making claims.
  Repeated OOS folds are not independent; seed dispersion is not a confidence interval.
- **Regimes:** a three-year training window and fixed parameters may fail after a
  regime change. No significance, capacity, or superiority claim is made.
- **Data vintage and survivorship:** Yahoo adjusted bars are not point-in-time
  vendor snapshots. Corporate-action adjustments, revisions, and today's selected
  surviving assets can introduce retrospective information. Caches preserve a
  download, not historical availability. Causal feature tests cannot detect these
  vendor/universe effects. The latest cache can differ from an earlier download.
- **Simplified execution:** fills use adjusted opens plus constant linear fees and
  slippage. Real opening-auction access, spreads, latency, partial fills, variable
  liquidity, taxes, and exchange outages are absent. Target exposure changes
  approximate traded notional: drift-driven rebalancing is not charged, and final
  exit costs use start-of-interval NAV. Short borrow fees, funding, cash interest,
  margin requirements, and short availability are absent. Constant slippage may
  understate costs, especially for crypto or stressed markets. There is **no market impact**.
- **Calendars:** intersecting calendars can condition the sample on missing-data
  availability. Crypto weekend risk remains in multi-day open returns, but weekend
  trading is disallowed. Policies do not exploit synchronized multi-asset execution.
- **Learning:** risk reward has trailing history beyond the provided observation;
  the observation is not a complete Markov state. PPO trains on repeated historical
  episodes, uses no financing model, and is not expected to learn reliably in the
  smoke budget. Full runs are longer but still do not establish predictive skill.

The code prevents designed feature/label/scaler leakage. Remaining future-information
risks are vendor revisions/adjustments, retrospective universe/calendar selection,
and human tuning after seeing OOS results. Treat any modification motivated by
these results as a new research hypothesis requiring fresh evaluation.


<!-- FINAL_RESEARCH_RESULTS -->
## Full-budget results and robustness checks

We repeatedly learned from earlier three-year windows and tested on the following up-to-six months, using 17 test windows per asset, five random starting points, and a trading charge of 7 basis points (0.07%) per unit of exposure changed.
PPO performed worse than every baseline on every asset by both average annual return after costs and Sharpe, which measures average return relative to its variability.
The numbers of PPO runs with positive annual returns were BTC-USD: 1 of 5; QQQ: 0 of 5; SPY: 0 of 5, so losses were consistent across starting points for SPY and QQQ.
The ± values measure variation across starting points rather than uncertainty about future markets, while the simple rules have zero variation because they are repeated unchanged.
Keeping decisions fixed, increasing fees plus slippage from zero to 20 basis points worsened PPO's average Sharpe on all three assets, and even free trading left it behind buy-and-hold on this measure.
At the separate, equal 1,024-step training budget, scrambling the training answers barely changed PPO's results, providing little evidence that it learned useful daily return timing at that budget.
The 1,000 random-agent comparisons per asset showed that a PPO starting point could rank well against luck while still losing money, so one favorable rank does not establish reliable skill.
We did not tune using these test results, and revised data, asset selection, simplified trading assumptions, and future market changes still limit the conclusion.

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

Read [the beginner guide](docs/EXPLAINED.md), [the full results discussion](docs/RESULTS_SUMMARY.md), and [the robustness report](results/checks/report.md). Run `python -m src.checks.run_checks`, then `python -m src.checks.estimate_runtime`, then `python -m src.checks.run_final`. Full metrics are in `results/final/metrics.csv`.

### Additional limitations

The five seeds share the same history; baseline seed spreads are zero by construction. Shuffled labels preserve average returns, and five permutations per fold do not establish statistical significance. Cost sensitivity holds policies fixed. Random-policy matching uses OOS positions for retrospective calibration, so percentile ranks are descriptive rather than formal probabilities of skill. Final outcomes are from already inspected dates and require a fresh holdout before further research claims.
<!-- END_FINAL_RESEARCH_RESULTS -->
