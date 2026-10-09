# IMPI — pre-registered hypotheses

**Registered:** {REG_DATE} (git commit recorded by P3+). Written after P2 (event counts, class rates and
power known) and BEFORE any hypothesis test, model comparison or test-set access. The test-set lock
`results/test_lock.json` did not exist at registration.
**Authors:** Team 23 — Fazil T, Faizur Rahman S, Suhail Ahmed S. Supervisor: Dr. Jeyavim Sherin RC.
**Frozen inputs:** feature library v1.0 (`results/metrics/feature_freeze.json`), configs in `config/`.

## 1. Setting

- Market: NSE equities. Data 2014-01-01 → 2026-10-09 (frozen). Official NSE files only.
- Splits (chronological, purged): train 2015-06-01 → 2020-12-31 · validation 2021-01-01 → 2022-12-31 ·
  test 2023-01-01 → 2026-10-09. The test set spans T+1 settlement and the 2024–25 F&O changes.
- Universe: **Tier 1** = stocks with stock futures on that day (point-in-time F&O list); **Tier 2** = EQ
  stocks with 60-day median turnover > ₹5 crore (secondary).
- Event-level exclusions: F&O ban on T or T+1; unadjusted close < ₹20; < 30 trading days since listing;
  < 252 days of history; ±3 trading days around a results meeting known by the T decision point.
  ASM/GSM cannot be applied historically (no point-in-time lists; `decisions.md` D-009).
- Disclosure: the team's earlier *Delivery Lens* pilot explored delivery behaviour on sharp rises during
  28 Jul 2025 – 8 Oct 2026, inside this test period. Every confirmatory result is therefore also reported
  excluding events dated on/after 28 Jul 2025.

## 2. Events and the primary endpoint

- **STRICT (headline):** algorithmic VCP — 2–4 contractions inside a 5–15-week base, strictly declining
  depth (each < 50%) and volume — and a close > pivot × 1.001 on volume ≥ 1.5 × its 20-day average.
- **BROAD (ML training set):** close above the previous 20-day high after a compression regime
  (Bollinger-bandwidth percentile < 30 on ≥ 5 of the previous 10 days).
- **B1 (ladder only):** every close above the previous 20-day high. Cooldown 10 trading days per family.
- **Primary endpoint:** triple-barrier success within 10 trading days: entry at the T+1 open, upper
  barrier entry + 2.0 × ATR14(T), lower barrier entry − 1.0 × ATR14(T); success = upper touched first;
  same-day double touch = failure. 20- and 60-day versions and volatility expansion
  (RV10 after / RV10 before ≥ 1.5) are secondary.

Counts at registration (Tier 1, eligible, labelled; from `p2_summary.json`):

{COUNTS_TABLE}

Class rates (share of successes): {CLASS_RATES}

Power (80%, two-sided α = 0.05): {POWER_TABLE}

Pre-registered power rule: if the STRICT Tier-1 test set cannot detect a 10-percentage-point H1
difference, H1 is additionally tested on STRICT events of Tier 1 OR Tier 2 (all liquid stocks) as a
secondary analysis. Standards are never lowered. Triggered at registration: **{POWER_TRIGGERED}**.

**Statement on STRICT power (decisions.md D-041).** The spec's VCP definition is rare in Indian F&O
stocks (far below the spec's expected 2,000–5,000 events, for every reasonable pivot rule). The STRICT
definition is kept exactly and all STRICT tests are run and reported, but groups with fewer than 30
events receive no p-value. The **BROAD** family is therefore expected to carry the inferential weight
for H1–H3, and H4 is evaluated on BROAD events as the spec intends. This statement is made before any
hypothesis test or outcome comparison was run.

## 3. Hypotheses (directional) and tests

**H1 — delivery expansion during the base → higher breakout success.**
Exposure: `deliv_z20` at T−1 > 0 (20-day mean delivery % above its own trailing-year norm, trigger day
excluded). Test: two-proportion z-test of success rates, exposed vs not. Supported if the exposed rate
is higher and the BH-adjusted q ≤ 0.10. Supporting model: logistic regression of success on exposure,
stock-level positioning alignment, their **interaction**, log 60-day turnover (size proxy), VIX
percentile, Amihud liquidity tercile and sector fixed effects, month-clustered standard errors.

**H2 — price rising while positioning falls (divergence) → lower follow-through.**
Exposure: 20-day return > 0 AND the stock's futures OI fell over the same 20 days (a short-covering-led
rise). Test: two-proportion z-test. Supported if the exposed rate is LOWER and q ≤ 0.10. Supporting
logistic model with the divergence × delivery interaction and the same controls.

**H3 — breakouts during FII selling with DII buying → favourable return asymmetry.**
Exposure (market-wide, D-003): 20-day change in FII net-long share of futures OI < 0 AND the same for
DIIs > 0. Metric: mean 10-day MFE / mean |10-day MAE|. Test: calendar-month block bootstrap
(1,000 resamples) of the exposed-minus-rest difference. Supported if higher and q ≤ 0.10. FII-only and
DII-only versions are always reported alongside (never only pooled).

**H4 — XGBoost on the same features beats the rule-based composite.**
Population: BROAD Tier-1 test events. (a) AUC, paired DeLong test; (b) top-decile precision (decile
thresholds fixed on validation), month-block bootstrap of the difference. Win = (a) supported (higher
AUC, q ≤ 0.10); loss = significantly lower; otherwise tie. Reported honestly whichever it is.

**Novelty criterion (ML):** the pairs (deliv_z20, FII stock-futures COT), (deliv_z20, stock OI COT) and
(deliv_z20, Bollinger-bandwidth percentile) are each checked for the top quartile of all feature pairs by
mean |SHAP interaction| (validation sample). Top quartile = direct support for the interaction claim.

## 4. Multiple testing and decision rules

- Exploratory family (P3, train + validation): H1, H2, H3 × {STRICT, BROAD} = 6 tests, BH q = 0.10.
- Confirmatory family (P6, test, run once): the same 6 + H4-AUC + H4-top-decile = 8 tests, BH q = 0.10.
- Any compared group with < 30 events gets no p-value and cannot be supported.
- Rule-based composite: fixed weights 0.30 compression + 0.25 delivery + 0.25 positioning + 0.20 volume;
  tradeable rule Score ≥ 1.0; standardisation from training events only; never tuned.
- Secondary, outside the families: 20/60-day horizons, volatility expansion, Tier-2 replication,
  benchmark ladder B0→B4 with costed backtests (deflated Sharpe; Hansen SPA / White Reality Check over
  every variant tried), walk-forward refits, sub-periods, COVID exclusion, placebo, VCP grid,
  104-week COT. Reversal and lead-lag analyses are **exploratory** and labelled as such.
- Negative results are reported in full.
