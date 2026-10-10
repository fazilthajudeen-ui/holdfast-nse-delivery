# IMPI — decisions log

Every place where the master spec was ambiguous, impossible with public data, or needed a concrete
rule. Rule from the spec: *choose the more conservative (anti-look-ahead) option and record it here.*
Dates are when the decision was made. "Verified" means checked against real NSE files on that date.

## Data availability and sources

**D-001 · Positioning is stock-level OI + market-level participant regime** (10 Oct 2026, decided by the user)
NSE's participant-wise OI file (`fao_participant_oi_DDMMYYYY.csv`) has only five rows — Client, DII,
FII, Pro, TOTAL — for the whole market (verified 2014, 2015, 2026). There is **no per-stock participant
breakdown** in any public NSE file. So: stock-level positioning comes from the F&O bhavcopy (futures
OI in shares, its change, a COT-style 0–100 index of OI level, price–OI build-up quadrants, futures
basis, option put/call OI); FII/DII/Pro/Client positioning is a market-wide regime/filter feature.

**D-002 · H2 and H3 reworded to match D-001** (10 Oct 2026)
H2: "price rising while the stock's futures OI falls (short-covering-led rise) → lower follow-through".
H3: "breakouts during market-wide FII-selling / DII-buying periods → favourable return asymmetry".
Final wording is in `pre_registered_hypotheses.md`.

**D-003 · No historical daily FII/DII cash-flow series** (10 Oct 2026, verified)
`fiidiiTradeReact` returns the current day only; `cat_turnover_DDMMYY.xls` (category-wise cash turnover)
has BNK/DFI/PRO/OTHERS — FIIs and most DIIs are inside OTHERS — and is not published for 2026.
Consequences: (a) the spec's "DII signal = cash flows PRIMARY" cannot be met; (b) the 20-day FII/DII
"flow" in the regime features and in H3 is proxied by the 20-day change in each one's net-long share
of (index + stock) futures OI; (c) the hedging-correction composite (cash + derivatives) is not built.
NSDL FPI data were not used (form-based archive; FII side only). Stated as a limitation.

**D-004 · Participant OI normalised as a share of total OI** (10 Oct 2026)
Positioning changes use contract counts (spec), but SEBI/NSE revise lot sizes (e.g. the Nov 2024
index-contract-size increase), which makes raw contract counts jump. Each participant's net long is
divided by the TOTAL row's OI for the same segment; the ratio is invariant to lot-size revisions
(unit test `test_market_positioning_share_is_lot_size_invariant`). Stock-level OI is in shares and
split-adjusted, so it is unaffected by lot-size changes.

**D-005 · Two equity-file eras, proven identical on overlap** (10 Oct 2026, verified)
`sec_bhavdata_full` is valid from 1 Oct 2019 (404 for 2014/2017/Jan 2019; for 30 Sep 2019 NSE served a
file dated 27 Jun 2019 — rejected by the internal-date check). Before that: legacy `cm…bhav.csv.zip`
(prices, volume, value, trades) joined with `MTO_DDMMYYYY.DAT` (deliverable quantity) on SYMBOL+SERIES.
On 1 Oct 2019 all 1,942 common rows matched exactly on open/high/low/close/prev-close/volume/trades/
delivery qty/delivery %. Twelve more overlap dates are cross-checked in the P0 QC report. Delivery from
MTO is kept only when MTO's traded quantity equals the bhavcopy volume.

**D-006 · Corporate actions: NSE records confirmed by the price gap** (10 Oct 2026)
NSE corporate-action records give the expected factor (split 10→2 = 0.2; bonus a:b = b/(a+b)); a record
is applied only when the opening gap open_T/close_(T−1) on the ex-date (or the next 2 trading days)
is within ±20% of it. Demerger/scheme records: applied with the observed gap when it is < 0.97 (e.g. Reliance → Jio Financial, Jul 2023, removed ~9% — a 0.90 cut-off would miss it).
Gaps < 0.6 or > 1.6 without any record within 5 trading days: snapped to a standard ratio within 3%,
otherwise logged and NOT adjusted (genuine crashes stay crashes). If > 5% of stocks gap on one day it
is a missed session, not corporate actions. Bonuses of preference shares / NCRPS / debentures are
ignored (equity share count unchanged). Dividends are not adjusted (price returns). Prices before an
adjustment are multiplied, quantities (volume, delivery qty, F&O OI in shares) divided (G6).
Logs: `results/metrics/corporate_actions_applied.csv`, `…_not_applied.csv`.

**D-007 · Size and liquidity without point-in-time market cap** (10 Oct 2026)
Bhavcopies carry no shares outstanding; AMFI's large/mid/small lists start only in 2018. Size control =
log of 60-day median rupee turnover; slippage buckets = point-in-time terciles of that turnover among
Tier-1 stocks each day (large/mid/small = 5/10/20 bps); the regression "liquidity bucket" control =
terciles of 20-day Amihud illiquidity (cut points from training events). Tier-2-only events have no
Tier-1 bucket and are charged the "small" slippage (conservative).

**D-008 · Sector = today's NSE industry label** (10 Oct 2026)
No point-in-time sector history is published. Industry labels from the current Nifty Total Market /
Nifty 500 constituent lists; stocks not in them (incl. most delisted names) are "Unknown". Minor bias,
stated as a limitation. Used only as fixed effects in the hypothesis regressions.

**D-009 · ASM/GSM exclusion cannot be applied historically** (10 Oct 2026)
NSE publishes only the current ASM/GSM lists. Applying today's list to the past would add look-ahead,
so the exclusion is not applied; reported as a limitation. The F&O ban exclusion IS applied (historical
ban files exist from 2014).

**D-010 · Data end date frozen at 9 Oct 2026** (10 Oct 2026)
So the test set cannot grow after P0. The Delivery Lens nightly job keeps running separately.

**D-011 · Pilot-window disclosure** (10 Oct 2026)
The Delivery Lens project already explored delivery behaviour on sharp rises for 28 Jul 2025 – 8 Oct
2026, which lies inside IMPI's test period. P6 therefore reports every confirmatory test twice: on the
full test period and excluding events dated on/after 28 Jul 2025.

**D-012 · Earnings exclusion uses only what was known** (10 Oct 2026)
±3 trading days around a results board meeting. A meeting AFTER day T excludes the event only if its
intimation timestamp (`bm_timestamp`) was at or before T 23:59 (G1). Meetings on non-trading days map
to the next trading day.

**D-033 · Release times and the decision point** (10 Oct 2026)
`data/release_lag.yaml`: every daily dataset is public by ~20:00 IST on T; decision point T 23:59;
entry at T+1 open. The F&O ban file dated D (bans for trade date D) is published the previous trading
evening. Features carry `available_at` = T 20:00; the builder asserts it never exceeds the decision point.

**D-034 · Special sessions** — Muhurat, Budget-day and NSE live-DR weekend sessions are trading days
(listed in `config/data.yaml`); the QC report flags any date where most stocks' file PREV_CLOSE does not
match our previous close (a missed session).

**D-035 · India VIX history** — 2014 `ind_close_all` files have no VIX row; NSE's
`historicalOR/vixhistory` API fills those dates (index-file values take precedence where both exist).

**D-039 · Symbol changes** — NSE `symbolchange.csv` is applied backwards in date order (A→B→C ends at
C); a change is skipped and logged if the new symbol already had rows before the change date.

## Features and events

**D-013 · Delivery z-score** — "20d delivery% z-score (μ, σ over trailing 252d)" = z-score of the 20-day
mean delivery % against the mean/sd of that same 20-day mean over the previous 252 days (window ending
T−1, ≥126 observations). H1's exposure uses the value at T−1 (formation window, trigger day excluded).

**D-014 · VCP detector interpretation**
* Fractal pivot high: high[i] > max of the k bars before and ≥ max of the k bars after (mirror for
  lows); confirmed only k bars later, so a pattern for day t uses pivots with i + k ≤ t − 1.
* Base: the highest confirmed pivot high within t−75..t−1 starts the base; base length ≥ 25 bars.
  After it, consecutive same-type pivots are merged (highest high / lowest low); the sequence must
  alternate H,L,H,L… and END on a low; n = number of (H,L) contractions, 2 ≤ n ≤ 4.
* "each depth < 50% of base height" is read as each contraction's depth (H−L)/H < 50% (the literal
  "50% of the base's high–low range" is impossible because the first contraction usually IS the range).
* Depths and average contraction volumes strictly decline; pivot = last contraction high; void if any
  close after it (before t) is already above pivot × 1.001.
* Trigger: close_t > pivot × 1.001 and volume_t ≥ 1.5 × mean(volume t−20..t−1).

**D-015 · Cooldown** — 10 trading days per stock per event family, applied to raw triggers (so the second
day of the same breakout is never a new event).

**D-032 · Tier 2** — EQ series, 60-day median turnover > ₹5 crore (point-in-time). The "Tier-2
replication" uses Tier-2 stocks that are NOT in F&O on that day.

**D-036 · Options in participant data** — v1.0 uses participant futures net positions only; participant
options long/short counts cannot be signed into a directional exposure without premium/delta data.

## Labels, splits and statistics

**D-016 · Same-bar double touch = loss** (label 0). **D-017 · Delisted/suspended** within the horizon →
label 0, exit at the last close; data simply ending → incomplete, excluded. Gap through a barrier fills
at the open.

**D-018 · Purging, CV and early stopping** — an event belongs to a split only if its whole label window
lies inside it. Hyper-parameters: purged 5-fold CV (contiguous time folds, purge overlapping label
windows, 10-trading-day embargo) on train+validation. Final models: fit on train, XGBoost early-stopped
on validation PR-AUC, isotonic calibration fitted on validation. Test untouched until P6.

**D-019 · Composite definitions** — CompressionQuality = z(mean(−z bbw_pct, −z atr_ratio, +z nr7_count,
−z range_contraction)); DeliveryZ = z(deliv_z20); PositioningAlignment = z(½·z(sign(ret20)·oi_chg20) +
½·(FII stock-futures COT − 50)/28.87); VolumeAnomaly = z(vol_z). All z's use TRAIN-split Tier-1 BROAD
events only, clipped to ±3; missing components = 0. Tier 2 (no positioning) re-weights 0.40/0.333/0.267.

**D-020 · Costs** (verified 10 Oct 2026; Budget 2026 STT hike was F&O-only) — STT 0.1% each side,
exchange 0.00297%, SEBI ₹10/cr, IPFT ₹10/cr, stamp 0.015% buy, GST 18% on brokerage+exchange+SEBI,
brokerage ₹20/order (spec default), DP charge ₹15.93 per sell (added; conservative), capital ₹10 lakh.

**D-021 · Benchmark** = Nifty 500 price index (TRI not in the daily files); excess return = stock return
from T+1 open to exit close minus the index return over the same dates.

**D-022 · Idle cash earns the risk-free rate** (6%/yr assumption, `costs.yaml`) and Sharpe/Sortino use
returns in excess of it — otherwise a strategy that is mostly in cash gets a meaningless negative Sharpe.

**D-023 · Ladder comparisons** — each rung is compared with the previous rung's events that the new filter
EXCLUDES. B4 = STRICT (or BROAD) events whose raw XGBoost score ≥ the 90th percentile of validation
BROAD scores. B0 = all eligible Tier-1 stock-days.

**D-024 · Small samples** — any compared group with < 30 events gets no p-value ("insufficient data") and
cannot be "supported"; backtests with < 30 trades are flagged.

**D-025 · Bootstrap** — resamples whole calendar months (keeps cross-sectional and short-run serial
dependence of overlapping events); p = 2(k+1)/(B+1), never exactly 0.

**D-026 · H3 metric** — return asymmetry = mean MFE_10 / mean |MAE_10|; regime = fii_flow20 < 0 and
dii_flow20 > 0 (D-003 proxies). FII-only and DII-only versions are reported separately (never only pooled).

**D-027 · SHAP novelty check** — pairs (deliv_z20, fii_fut_stk_cot), (deliv_z20, oi_cot),
(deliv_z20, bbw_pct252); "top quartile" = rank ≤ ¼ of all pairs by mean |interaction| AND non-zero
(many pairs tie at exactly zero).

**D-028 · Deflated Sharpe** — N = every row of `results_log.csv` (all variants ever tried, conservative);
variance of trial Sharpe ratios from all logged backtests.

**D-037 · Multiple testing families** — exploratory (P3, train+validation): H1–H3 × {STRICT, BROAD} = 6
tests. Confirmatory (P6, test): those 6 + H4-AUC + H4-top-decile = 8 tests. BH at q = 0.10 within each.

## Engineering

**D-029 · PIT store partitioned by year** — one Parquet file per dataset per year with a `date` and an
`available_at` column on every row (daily partitions = ~3,200 tiny files per dataset).

**D-030 · Gates enforced mechanically** (10 Oct 2026) — the user asked for the whole build in one go, so
the human review gates became code guards: P3+ refuse to run unless `pre_registered_hypotheses.md` is
committed in git and unchanged; P3+ refuse if the frozen feature library (hashes of configs, feature
code and FEATURES_V1) changed after P2; P6 refuses if the frozen models changed after P5, and runs once
(`results/test_lock.json`; a rerun is allowed only if a previous attempt crashed before writing results).

**D-031 · Windows Application Control** blocks two compiled modules on this laptop (`pyarrow.dataset`,
statsmodels' state-space smoother). The code avoids them (pandas filtering; GLM imported directly). No
effect on results; security settings were not changed.

**D-038 · Sandbox rehearsal** — the full pipeline was rehearsed end-to-end on synthetic data
(`tests/integration/make_synthetic.py`) with outputs redirected away from `results/`; the sandbox switch
works only when both output folders are redirected, so it can never write real results.

**D-040 · Internally inconsistent delivery records are treated as missing** (10 Oct 2026, verified)
NSE's `sec_bhavdata_full` for 11 Sep 2026 publishes DELIV_PER values that disagree with
DELIV_QTY / TTL_TRD_QNTY for 98% of stocks (median 57.3% vs 48.6%; the previous day agrees exactly), and
a few rows of the 21 Oct 2025 Muhurat session do the same. Which field is wrong cannot be known, so:
a row whose two delivery figures differ by more than 0.5 percentage point is blanked, and a day where
more than half of EQ rows are inconsistent has all its delivery blanked. Counts are in the PIT build
report. (The Delivery Lens project reads the same file and is affected on that date.)

**D-041 · The spec's VCP is rare; it is NOT loosened** (10 Oct 2026, counts only — no outcomes looked at)
Early count on contiguous data (Tier-1 STRICT triggers, Jan 2015 – Jul 2016, before exclusions): spec
k=2 → 9; k=3 → 22; k=5 → 51; a causal 5% zigzag pivot → 24. Scaled to 11 years that is ~65–370, far
below the spec's expected 2,000–5,000 — for every reasonable pivot rule, so the rarity comes from the
pattern conditions themselves (2–4 contractions, strictly declining depth AND volume), not from noisy
pivots. Following the spec ("if insufficient, widen to Tier 2, don't lower standards"): the STRICT
definition is kept exactly; the pre-registered power rule adds Tier-2 STRICT; STRICT results are
reported in full but are expected to be underpowered, so the BROAD family (compression breakouts)
carries the inferential weight. This was written into the pre-registration before any test was run.
The P7 grid (k = 2/3/5 × depth × volume × base length) reports how counts move with the parameters.

**D-006 (refined on the full data, 10 Oct 2026)** — (a) gap-only snaps are applied only for DOWNWARD
standard ratios (splits/bonuses) and only when the stock also traded on the previous market day: 367
unexplained overnight "doublings" were illiquid stocks reopening after suspensions, not consolidations
(NSE lists only 2 consolidations in 12 years), so they are left as real moves; (b) for demerger records,
when the close-to-close ratio is far below the open gap (< 0.8×), the ex-date OPEN in old bhavcopies is a
stale pre-open band price and the close-to-close ratio is used (ADANIENT 2015-06-03: open 0.90×, close
0.17×); (c) abbreviated NSE subjects such as "Fv Splt Frm Rs 10 To Rs 2" are parsed (ICIL 2016-11-11).
Final: 1,442 adjustments (840 NSE-confirmed, 468 gap-snapped splits/bonuses missing from NSE's API — all
five on F&O stocks verified genuine, e.g. Bajaj Finance 2025-06-16, JSW Steel 2017-01-04 —, 133
demergers); 622 logged and not adjusted. Remaining |daily move| > 30% on F&O stock-days: 55, all
checked to be genuine events (Yes Bank 2020, DHFL, Jet Airways, Vodafone Idea, Infibeam 2018, PSU-bank
recapitalisation rally 2017-10-25, PolicyBazaar 2026-09-24).

**D-042 · Regime features computed per index series** (10 Oct 2026, before registration)
On the 14 days NSE's `ind_close_all` file is missing, the VIX history API still adds a row, so a joint
pivot left gaps in the Nifty series and voided every 200-day window for 200 rows (30% of events lost
`nifty_above_200`). Each index series is now computed on its own dates and re-indexed. Fixed and
re-frozen before the pre-registration commit (15103f1); no test had been run.

**D-043 · Outcome record** (10 Oct 2026)
Pre-registration commit 15103f1 (08:54 IST); one test-set run, lock commit ea49def (09:23–09:29 IST).
No pre-registered hypothesis was supported; H4 was a significant loss for the ML challenger. Results
are reported in full in `results/summary.md` and the paper draft.
