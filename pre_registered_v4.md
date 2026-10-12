# IMPI v4 — pre-registration (re-framed delivery, divergence, COT and loose-VCP hypotheses)

Written and committed on 12 Oct 2026, **before** any association between these signals and any outcome
was computed. Up to this commit the signals were only built (`src/impi/v4/signals.py`) and counted
(sample sizes below). Nothing in v4 is trained: every signal is a fixed rule.

## Why v4

The pre-registered v1 study (committed 10 Oct 2026, `15103f1`) supported none of H1–H4. Reviewing it, the
team raised a point the v1 design missed: **delivery is direction-neutral**. Every delivered share has a
seller who delivered it, so high delivery shows that shares really changed hands (conviction,
participation), not who was in control — it can be accumulation *or* distribution. v1's H1 assumed
delivery expansion is bullish. v4 tests delivery as a predictor of **move size** (D1) and, using price
direction to separate buying from selling, of **direction** (D2). It also re-tests the best v1 lead (H2)
with more events and a 20-day horizon, tests positioning the way Larry Williams used the US COT report
(market level, contrarian extremes, multi-week horizon — C1), and tests a looser VCP rule (V1).

v1 results are unchanged and stay reported as they are.

## What has already been seen (disclosure)

- The 2023–2026 period was used once for the v1 test (and post-hoc for v2/v3). v4's rules have never been
  evaluated on it, but it is not a clean period. Results there are reported as **"rules fixed in advance,
  previously used period"**. The **confirmatory** evidence is the live forward record (below).
- C1 follows an exploratory v1 lead (train+validation only): when FII stock-futures COT was in its bottom
  decile, the next 60-day Nifty 500 return averaged +6.4% vs 2.9% overall.
- V1's rule was chosen from the v1 robustness grid, which used **training + validation data only**
  (2015–2022): among k = 5 settings, the one with the most events at the specification's middle depth
  (k = 5, max depth 0.50, volume ×1.25, base 20–60 days; 417 train+val events, H1 difference +8.9 pp there).
- The v2/v3 models include delivery features, but the marginal association of delivery with move size,
  and the signed-delivery score, have never been computed.

## Hypotheses (all fixed)

| # | Claim | Signal (known by T 23:59) | Outcome | Statistic | Expected |
|---|---|---|---|---|---|
| **D1** | High delivery precedes bigger moves, either direction | `deliv_z20` (5-day delivery vs its own past, z-score) | `move20` = \|close(T+20)/open(T+1) − 1\| ÷ (rv20 · √(20/252)) | mean over dates of the daily cross-sectional Spearman IC | > 0 |
| **D2** | Signed delivery predicts direction (accumulation vs distribution) | `sds` = mean over the last 10 days of (delivery % − previous-60-day median) × sign(daily price change) | `fwd_20` = close(T+20)/open(T+1) − 1 | mean daily Spearman IC | > 0 |
| **H2′** | Breakouts with price up while futures OI falls follow through less | all 20-day-high breakouts (B1), Tier 1, eligible; exposed if ret_20 > 0 and oi_chg20 < 0 | `label_20` (triple barrier +2/−1 ATR, 20 days) | success(exposed) − success(rest), two-proportion z | < 0 |
| **C1** | Williams-style contrarian COT: extreme FII stock-futures pessimism precedes market gains | FII stock-futures COT index ≤ 10 (bottom of its 52-week range) | Nifty 500 close(T+61)/close(T+1) − 1 | mean outcome on signal days − mean on all days | > 0 |
| **V1** | In loose-VCP breakouts, delivery expansion (accumulation into a breakout) succeeds more | loose VCP (rule above), Tier 1, eligible; exposed if deliv_z20_prev > 0 | `label_10` (triple barrier +2/−1 ATR, 10 days) | success(exposed) − success(rest), two-proportion z | > 0 |

Universe for D1, D2: eligible Tier-1 (F&O) stock-days, dates with ≥ 30 stocks having both values.

**Inference.** D1, D2: circular block bootstrap over dates (block 20 trading days, 2,000 draws).
C1: circular block bootstrap of the daily series (block 60 trading days, 2,000 draws). H2′, V1: two-sided
two-proportion z (supporting: month-block bootstrap). All p-values two-sided. Random seed 20261012.

**Pass rule.** A hypothesis is supported only if its estimate has the expected sign **and** its
Benjamini–Hochberg q ≤ 0.10 within its family (the five tests above). Minimum sample: H2′ and V1 need
≥ 30 events in each group, C1 needs ≥ 30 signal days with an outcome; otherwise "insufficient data".

## Evaluation 1 — 2023–2026 check (run once)

Signal dates 1 Jan 2023 – 9 Oct 2026 with a complete outcome window, research data (frozen 10 Oct 2026).
Run once by `python -m impi.v4.check`, guarded by `results/v4_check_lock.json`.
Sample sizes (counted before this commit): D1/D2 160,735 stock-days over 935 dates; H2′ 4,277 breakouts
(2,271 exposed / 2,006 rest; minimum detectable difference ≈ 3.9 pp at 80% power); V1 211 events
(137 / 74; minimum detectable ≈ 17.5 pp — **underpowered**); C1 145 signal days of 935 (clustered, so few
independent episodes).

## Evaluation 2 — live forward record (confirmatory)

From the first live trading day after 9 Oct 2026, the nightly run logs every signal **before** its outcome
exists (`data/track/predictions_v4.csv`, `predictions_v4_market.csv`). Each hypothesis is evaluated once,
when its pre-set minimum is reached (interim numbers before that are descriptive only):

| Test | Final evaluation when | Expected |
|---|---|---|
| D1, D2 | 60 prediction dates with matured 20-day outcomes | ~Feb 2027 |
| H2′ | 1,000 matured breakouts (~970 per year) | ~Oct–Nov 2027 |
| V1 | 30 matured events in each group (~54 events per year) | ~2028 |
| C1 | 750 trading days with matured outcomes | ~2029 |

Same statistics, pass rule and BH family (over the tests evaluated at that time).

## Display (not part of any test)

The terminal highlights, for information only: accumulation / distribution days (D2 score in the top /
bottom 10% of that day's F&O stocks), "big move watch" (deliv_z20 ≥ 2), loose-VCP breakouts (V1),
divergent breakouts (H2′ exposure), and the C1 market state. These display thresholds are not tested
and are not trading signals.

Research tool for education — not investment advice.
