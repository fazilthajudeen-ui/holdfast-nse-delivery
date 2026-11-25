# IMPI v3 — pre-registration (option-implied features + 20-day targets)

Written and committed on 11 Oct 2026, **before** any v3 model was trained or any association between the
new features and any outcome was computed. Up to this commit, the option data were only parsed, solved
for implied volatility and checked for quality (round-trip tests, the legacy-vs-UDiFF overlap check,
coverage and value ranges).

## Why v3, and what it cannot claim

The pre-registered v1 study is finished: no hypothesis was supported, and its 2023–2026 test set has
been used once. v2 (post-hoc) found one useful model, volatility expansion (CV AUC 0.742, 2023+ 0.743),
plus a small, untradable edge for beating the median F&O stock. v3 adds information v2 did not have:

1. **Option-implied features** from the F&O bhavcopy (already downloaded; no new data source).
2. **20-day targets**, because the Delivery Lens pilot's only significant effect was at 20 days.

Every 2023+ number for v3 is **POST-HOC**, because that period was already used. The **only confirmatory
evidence** for v3 will come from the live forward track record (predictions logged before outcomes exist).

## Data and features (fixed)

- Source: NSE F&O bhavcopy, legacy format to 5 Jul 2024 and UDiFF from 8 Jul 2024. Implied volatilities
  are identical between the two formats on the overlap days 1 and 5 Jul 2024 (max difference 0.0).
- Only option contracts that **traded** that day are used (untraded contracts carry stale closes).
- Forward = the same-expiry stock-future close; Black-76, r = 6.5%, act/365; OTM side only (calls K ≥ F,
  puts K < F); ATM IV by interpolation at log-moneyness 0; iv30 = 30-day constant-maturity ATM IV by
  total-variance interpolation (near expiry skipped within 3 days of expiry).
- 14 option features (`src/impi/v3/panel.py: OPT`): iv30, iv_rv, iv_rv_spread, iv_term, iv_skew, iv_smile,
  iv_chg5, iv_chg20, iv_pct252, iv_z60, dte_near, log_pcr_vol, log_opt_fut_vol, opt_vol_z60.
- Release T 19:30 ≤ decision point T 23:59 (G1). Truncation test: `tests/test_v3.py`.

## Targets (fixed)

| Target | Definition | Purge end |
|---|---|---|
| vol_exp | RV10(after) / RV10(before) ≥ 1.5 | label_end |
| label_xs10 | 10-day forward return beats the day's eligible Tier-1 median | label_end |
| label_10 | triple barrier +2/−1 ATR14, 10 days | label_end |
| label_xs20 | 20-day forward return (T+1 open → T+20 close) beats the day's median | label_end_20 |
| label_20 | triple barrier +2/−1 ATR14, 20 days | label_end_20 |

## Development protocol (fixed; no 2023+ data in any choice)

- Rows: eligible Tier-1 stock-days from 1 Jun 2015 whose label window ends by 31 Dec 2022.
- Walk-forward CV, test folds = calendar years 2018–2022, purged by each target's label end,
  15-calendar-day embargo (same as v2).
- Stage 1: feature sets `v1_raw`, `v1v2_raw` (v2 baselines), `v1v2opt_raw` (+ options) and
  `xs_rank_opt_regime` (cross-sectional ranks incl. options + regime), each with XGBoost config c1.
- Stage 2, on the best stage-1 set: XGBoost c2, XGBoost c3 (depth 6, 800 trees, lr 0.03, min_child_weight
  200, subsample 0.7, colsample 0.5, lambda 10), logistic regression, and the probability-average ensemble
  of the best XGBoost config with the logistic model.
- Chosen model per target = highest mean fold AUC. Refit on all development rows; isotonic calibration on
  2021–2022 out-of-fold scores. The pickled models are hashed (sha256) **before** the 2023+ scoring.
- **Adoption rule:** a v3 model replaces the v2 model of the same target in the terminal only if its CV mean
  AUC is ≥ v2's + 0.005 **and** higher in ≥ 4 of the 5 folds. New targets (20-day) have no v2 model.
  The 2023+ post-hoc result is reported whatever it shows and does **not** change adoption.
  Both v2 and v3 predictions keep being logged nightly, whichever is shown.
- No other variants. Any later change is a new version with its own pre-registration.

## Forward (confirmatory) tests on the live track record

Population: every Tier-1 F&O stock scored by the nightly live run, from the first night the frozen v3
models are logged. Predictions are appended to `data/track/predictions_v3.csv` before outcomes exist.

Evaluation date: the first night on which **60 distinct prediction dates** have matured outcomes for the
20-day targets (expected around Feb 2027). Before that, numbers are shown as interim and descriptive only.

| Test | Claim | Statistic |
|---|---|---|
| F1 | v3 volatility model ranks better than v2 | paired AUC(v3) − AUC(v2) > 0 on the same stock-days |
| F2 | v3 volatility model has skill | AUC(v3, vol_exp) > 0.5 |
| F3 | 20-day relative model has skill | AUC(v3, label_xs20) > 0.5 |
| F4 | 20-day breakout model has skill | AUC(v3, label_20) > 0.5 |
| F5 | 10-day relative model has skill | AUC(v3, label_xs10) > 0.5 |

All one-sided. Inference: bootstrap over **prediction dates** (2,000 draws; dates resampled with
replacement, because outcomes of one date are correlated and windows overlap). Benjamini–Hochberg FDR at
q = 0.10 across F1–F5. A test is supported only if its BH-adjusted q ≤ 0.10.

Secondary, descriptive only: calibration (Brier vs base rate), top-decile minus average 20-day return
net of 0.45% round-trip costs, AUC by month.

## What would count against v3

- F1 not supported: implied volatility does not improve the volatility model out of sample.
- F3/F4 not supported: the 20-day lead from the Delivery Lens pilot does not carry over.
These outcomes will be reported in full, exactly like the v1 null result.

Research tool for education — not investment advice.
