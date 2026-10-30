# Application material: honest options

PPO means Proximal Policy Optimization, an algorithm that learns actions from
numerical rewards. A seed fixes a random starting point; a basis point is 0.01%.
Net means after trading expenses. These three options describe the same project;
choose the wording that fits your application rather than presenting them as
three independent achievements.
Purging means dropping training answers known only at or after the next test starts.
A fit trains one model for one seed and window; the full budget requests 10,000
learning steps.

## Three resume bullet options

1. Built a Python trading-research pipeline for **3 assets**, using **17 chronological
   test windows**, **5 seeds**, and **7 basis points** of trading costs; documented
   original PPO's underperformance against buy-and-hold instead of claiming profitability.
2. Implemented and tested past-only inputs, purged training answers, and shared
   cost accounting with **34 passing tests**; audited **120 strategy histories** of
   **2,016 observations each** and froze **4 follow-up variants** in Git before fitting.
3. Evaluated **765 full-budget PPO seed/window fits** in the original and follow-up
   comparisons; retained all **4 proposed variants** and reported **0 of 12**
   asset/variant comparisons exceeding buy-and-hold's mean yearly return after costs.

The 765 count includes 255 original and 510 follow-up full-budget PPO fits, not the
smaller diagnostic or smoke fits. It does not count deterministic rule paths as
trained agents. The 120 histories include repeated deterministic seed paths;
they are not 120 independent market experiments.

## Four-sentence “what I learned” paragraph

I learned that correct information timing and trading-expense accounting matter
more than choosing a complicated learning algorithm.
I separated training from later evaluation, tested that future data could not
change earlier inputs, and compared multiple repeatable random starting points.
I learned to retain negative results and to distinguish lower trading activity
or lower risk from genuinely higher returns.
Because the follow-up reused already inspected dates, I would require new
untouched history before claiming that any improvement will last.

## Check the claims

The machine-readable values and their source files are in
[`results/application_facts.json`](../results/application_facts.json).
Original outcomes are in [`results/final/metrics.csv`](../results/final/metrics.csv);
all eight strategies are in
[`round_001/metrics.csv`](../results/improvement_round/round_001/metrics.csv).
The pre-run suite passed 32 tests; two archive-preservation tests brought the
publication suite to 34. No statement implies live trading, deployed capital,
market prediction skill, independent future evidence, or profitable PPO.
