# IMPI v5 — pre-registration (breakout strategy improvement study)

Written and committed on 14 Oct 2026, **before** any v5 strategy variant was backtested on any period.
Up to this commit the code (`src/impi/v5`) was only unit-tested on synthetic data and smoke-tested for
errors on Jan–Mar 2016 events (no performance figures were looked at), and event counts were made
(below). Nothing in v5 is trained: every variant is a fixed rule.

## Why v5

In v1 (pre-registered, test set opened once) breakouts traded with a +2/−1 ATR, 10-day exit lost money
after Indian costs (all breakouts CAGR −26.8%, BROAD −12.6%, 2023–26; Table IV). The team asked whether a
better-designed breakout strategy can do better. v5 answers that honestly: a fixed grid of standard
improvements (market-trend filter, stock-strength filter, longer exits), **one** variant chosen on
2015–2022 only by a fixed rule, then checked **once** on 2023–2026, then tracked live.

v1–v4 results are unchanged and stay reported as they are.

## What has already been seen (disclosure)

- 2023–2026 was used by v1 (test), v2/v3 (post-hoc) and v4 (check). Known from it: the v1 backtests
  (Table IV), breakout success falling by year (28.8%, 24.0%, 24.3%, 17.4%), and v4's V1 result (high
  delivery into a loose-VCP breakout was a warning sign). **v5 uses no delivery filter**, so that V1 finding
  cannot leak into the choice. The market and stock filters below are standard breakout-trading rules
  (trend, nearness to 52-week high, relative strength), not taken from 2023–26 results.
- The v5 check on 2023–26 is therefore **POST-HOC** ("rule chosen in advance, previously used period"),
  never a confirmatory test. The confirmatory evidence is the live forward record (below).

## The grid (120 variants, all fixed)

Universe: Tier-1 (F&O), eligible events (same exclusions as v1). Entry at the T+1 open; at most 20
equal-weight positions; full Indian costs and slippage (`config/costs.yaml`); idle cash at 6%; same-day
ties broken by a seeded random draw (no ranking).

| Part | Options |
|---|---|
| Event set | **B1** every 20-day-high breakout · **BROAD** B1 after compression (v1 main set) |
| Market filter | **M0** none · **M1** Nifty 50 above its 200-day average · **M2** Nifty 500 up over the last 20 days |
| Stock filter | **S0** none · **S1** close within 10% of its 52-week high · **S2** 120-day return in the top 30% of that day's Tier-1 stocks · **S3** S1 and S2 |
| Exit | **TB10 / TB20 / TB60** triple barrier +2/−1 ATR within 10/20/60 days · **T20 / T60** close of T+20 / T+60 |

All filters use data known by 23:59 on day T. Splits: train 1 Jun 2015 – 2020, validation 2021–22,
test 1 Jan 2023 – 9 Oct 2026; an event belongs to a split only if its **60-day** window ends inside it
(so all exits use the same events). The v1 rules B1|M0|S0|TB10 and BROAD|M0|S0|TB10 are in the grid and
are re-run in this frame as baselines.

## Selection rule (development, `python -m impi.v5.dev`)

Every variant is backtested separately on train and on validation. Among variants with **≥ 30 trades in
both**, the one with the highest **min(train Sharpe, validation Sharpe)** after costs is chosen (it must
work in both periods, not just one). Every variant is logged to `results/results_log.csv`. The 2023+
period is not read by this run.

## Evaluation 1 — 2023–2026 check (run once, `python -m impi.v5.check`, lock `results/v5_check_lock.json`)

The chosen variant (and the two v1 baselines, for comparison) are backtested once on the test split.
Reported: CAGR, Sharpe, max drawdown, profit factor, trades, success rate, yearly returns, Nifty 500 over
the same days.

**Pass rule** ("improvement supported", post-hoc): Sharpe above the Nifty 500 **and** Hansen SPA p ≤ 0.10
vs the Nifty 500 (stationary bootstrap, 1,000 reps, block 10) **and** deflated Sharpe ratio ≥ 0.95, with
the number of trials = all rows in `results/results_log.csv` at that time (v1–v5) and the trial-Sharpe
variance from the 120 v5 validation Sharpe ratios. Otherwise: "no reliable improvement". Seed 20261010
(+500 / +503).

Event counts (Tier 1, eligible, 60-day purge; train / validation / test): B1 4,825 / 1,744 / 4,103;
BROAD 2,016 / 820 / 1,710; smallest cell BROAD|M2|S3 493 / 174 / 366.

## Evaluation 2 — live forward record (confirmatory)

The chosen rule is frozen (its name and hash are in `results/metrics/v5_dev.json`). From the first live
trading day after the check, the nightly run logs the rule's new signals **before** their outcomes exist.
Final evaluation once, when there are **≥ 100 closed trades and ≥ 250 trading days** of record: the same
portfolio rules and pass rule (Sharpe above the Nifty 500 over the same days, SPA p ≤ 0.10). Interim
numbers before that are descriptive only.

Research tool for education — not investment advice.
