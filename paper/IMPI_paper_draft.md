# Delivery, Positioning and Compression: Do Their Interactions Predict Breakout Success on the National Stock Exchange of India?

*Draft for a conference submission — Team 23 (Fazil T, Faizur Rahman S, Suhail Ahmed S), Department of
Computer Science and Engineering (AIML), Hindustan Institute of Technology & Science. Supervisor:
Dr. Jeyavim Sherin RC.*

> Status: complete first draft (10 Oct 2026). All numbers come from `results/metrics/*.json` and
> `results/summary.md`; figures are in `results/figures/`. Items marked **[TEAM]** need the team's input
> (literature on Indian markets, venue formatting, final citation checks).

## Abstract

We test whether the interaction of delivery behaviour, positioning and price compression predicts
breakout success on the National Stock Exchange of India. Using only official NSE files for
1 January 2014 – 9 October 2026 (3,161 trading days; delivery data before October 2019 reconstructed
from the legacy bhavcopy and the MTO file and verified to match exactly on twelve overlap dates), we
build a point-in-time panel of F&O stocks, label breakouts with a triple barrier (two ATRs up, one
down, ten days) and pre-register four hypotheses before any test. On 4,876 compression breakouts
(train 2015–2020, validation 2021–2022, test 2023–2026, evaluated once), delivery expansion during the
base does not raise success (test: 24.8% vs 24.9%, p = 0.99); price rising while futures open interest
falls lowers success by 2.8 points, in the predicted direction but not significantly (q = 0.29); and
market-wide FII-selling/DII-buying regimes do not change return asymmetry. An XGBoost model on 80
frozen features is significantly worse than a fixed-weight rule (test AUC 0.460 vs 0.526,
DeLong q = 0.024), and walk-forward refits reach only 0.502. SHAP interaction values show no
delivery × positioning or delivery × compression interaction. Breakout strategies lose money after
Indian transaction costs, and Hansen's SPA test finds no variant that beats the Nifty 500. The
algorithmic volatility contraction pattern, as specified, is too rare in F&O stocks (82 events in
eleven years) for inference. We report these negative results in full, together with the data
pipeline, as a reproducible baseline for delivery-based research in India.

## 1. Introduction

Every evening the National Stock Exchange of India (NSE) publishes, for each security, the share of
the day's traded quantity that ended in delivery — shares actually taken into demat accounts rather
than bought and sold within the day. Indian market participants have long read high delivery as a sign
of "genuine" buying, yet the claim is rarely tested with the controls that modern empirical finance
requires: point-in-time universes, no look-ahead, transaction costs, and corrections for the many
variants an analyst inevitably tries.

Delivery is only one of three daily signals that are public, free and specific to the Indian market
structure. The second is positioning: open interest in single-stock futures and options (published per
stock) and the participant-wise open interest of foreign institutional investors (FIIs), domestic
institutions (DIIs), proprietary desks and clients (published only as market-wide totals). The third is
price compression — narrowing ranges and falling volatility before a breakout, popularised by
practitioners as the volatility contraction pattern (VCP; Minervini, 2013).

Each signal has a folk literature of its own. Our question is about their **interaction**: does a
breakout from a compressed base work better when buyers are taking delivery during the base, when
stock-level positioning confirms the move, and when institutional flows are supportive — beyond what
each signal tells us alone? We answer it with a pre-registered design over eleven years of official
NSE data, a strict chronological split with a test set evaluated exactly once, and a machine-learning
challenger whose SHAP interaction values directly measure whether delivery interacts with positioning
and compression.

**Contributions.** (i) A reproducible, point-in-time research dataset built only from official NSE
archives (2014–2026), including the reconstruction of pre-2019 delivery data from the legacy bhavcopy
and the MTO delivery file, and corporate-action adjustment of prices *and* quantities. (ii) A
pre-registered test of three directional hypotheses on delivery, positioning divergence and
institutional flow regimes, with false-discovery-rate control. (iii) A benchmark ladder from the
unconditional base rate to an ML-filtered strategy, evaluated after Indian transaction costs with
deflated Sharpe ratios and Hansen's SPA test. (iv) An honest account of what the public data cannot
support — notably that participant positioning is not published per stock.

## 2. Related work

**Positioning and the Commitments of Traders.** Studies of CFTC Commitments of Traders data examine
whether the net positions of trader categories anticipate futures returns (e.g., Wang, 2003; Sanders,
Boris and Manfredo, 2004). We borrow the COT-index normalisation (a position's location within its own
52-week range) and apply it to NSE's participant-wise open interest and to stock-level futures OI.

**Delivery and institutional flows in India.** **[TEAM]** Add peer-reviewed studies on delivery
volume, FII/DII flows and returns on Indian exchanges. Verify every citation against the source before
submission; do not cite from memory.

**Price patterns and technical analysis.** Lo, Mamaysky and Wang (2000) formalised the automatic
detection of chart patterns with statistical tests; Jegadeesh and Titman (1993) documented momentum.
The VCP is a practitioner construct (Minervini, 2013); we implement it as an explicit algorithm
(fractal pivots, declining contraction depth and volume) so that every event is reproducible.

**Machine learning in finance and its pitfalls.** Gu, Kelly and Xiu (2020) show that tree ensembles and
neural networks can improve return prediction. López de Prado (2018) describes the triple-barrier
labelling, purged K-fold cross-validation with embargo and the multiple-testing problem we follow.
We use XGBoost (Chen and Guestrin, 2016), random forests (Breiman, 2001), isotonic calibration
(Niculescu-Mizil and Caruana, 2005) and SHAP values, including TreeSHAP interaction values (Lundberg
and Lee, 2017; Lundberg et al., 2020). Strategy-level overfitting is addressed with White's (2000)
Reality Check, Hansen's (2005) SPA test and the deflated Sharpe ratio (Bailey and López de Prado, 2014).

## 3. Data

**Sources.** All data come from official NSE archives: the full bhavcopy with delivery
(`sec_bhavdata_full`, valid from 1 October 2019); before that, the legacy equity bhavcopy joined with
the security-wise delivery position file (MTO); F&O bhavcopies in the legacy format (to 5 July 2024) and
the UDiFF common format (from 8 July 2024); participant-wise open interest; the F&O ban list; daily index
closes (Nifty 50/200/500) and India VIX; corporate actions, results board-meeting dates with their
intimation timestamps, and symbol changes. Data run from 1 January 2014 (a one-year warm-up) to
9 October 2026, frozen before analysis.

**Validation of the reconstruction.** On every date where both equity formats exist, the legacy
reconstruction and `sec_bhavdata_full` agree exactly on prices, volumes and delivery quantities;
both F&O formats agree exactly on settlement prices and open interest on the overlap days
(`results/metrics/qc_report.json`). Files whose internal trade date differs from the requested date —
NSE serves a previous day's file on some holidays — are rejected.

**Adjustment.** Splits, bonuses, consolidations and demergers listed by NSE are applied only when the
observed opening gap confirms the expected ratio (±20%); prices before the ex-date are multiplied and
quantities (volume, delivery quantity, futures OI in shares) divided by the factor. Unlisted gaps are
adjusted only when they match a standard ratio within 3%; genuine crashes are left untouched.

**Universe.** Tier 1 consists of the stocks with single-stock futures on each day, reconstructed from
that day's F&O bhavcopy, so the universe is point-in-time and includes stocks that later left the
segment or were delisted. Tier 2 adds all EQ-series stocks with a 60-day median turnover above ₹5 crore.

**What the data cannot provide.** NSE does not publish participant-wise positions per stock, nor a
historical daily series of FII/DII cash-market flows, nor historical ASM/GSM surveillance lists. These
limits shape the design (Section 4) and are listed in Section 7.

## 4. Methodology

**No look-ahead.** Each dataset carries a release time (all daily files are public by about 20:00 IST
on day T). Features for day T use only data released by the T decision point and trades are entered at
the T+1 open. Rolling baselines end at T−1; pattern pivots are used only once confirmed; a future
results meeting excludes an event only if it had already been announced. A truncation test confirms on
real data that every feature for a date is unchanged when later data are added.

**Features (v1.0, frozen before testing).** *Delivery:* the z-score of the 20-day mean delivery
percentage against its own trailing-year history, and an accumulation variant active when the 20-day
price change is within ±5%. *Volume and liquidity:* the z-score of log adjusted volume, the up/down
volume ratio, and 20-day Amihud (2002) illiquidity. *Compression:* Bollinger-bandwidth percentile,
ATR14/ATR50, NR7 count and a range-contraction ratio. *VCP:* contraction count, first and last depth,
base length and the volume trend. *Stock positioning:* futures OI change, a 52-week COT-style index of
OI, its acceleration and persistence, price–OI build-up quadrants, divergence magnitude, annualised
futures basis and the put/call OI ratio. *Market positioning:* for FIIs, DIIs, proprietary traders and
clients, in index and stock futures separately, net long as a share of total OI (invariant to lot-size
revisions), its COT index, acceleration, persistence and divergence. *Regime:* Nifty trend (50- vs
200-day average), India VIX percentile, 20-day FII and DII positioning flows and market breadth.

**Events and labels.** STRICT events are algorithmic VCP breakouts; BROAD events are 20-day-high
breakouts after a compression regime. Each event is labelled with a triple barrier (López de Prado,
2018): entry at the next open, profit-take at two ATRs, stop at one ATR, ten trading days (20 and 60
as secondary horizons); the maximum favourable and adverse excursions are stored. A secondary label
marks volatility expansion.

**Hypothesis tests.** H1–H3 (Section 2 of the pre-registration) use two-proportion z-tests or a
calendar-month block bootstrap, supported by logistic regressions with size, liquidity, volatility-regime
and sector controls and the delivery × positioning interaction, with month-clustered errors. False
discovery is controlled with Benjamini and Hochberg (1995) at q = 0.10 within the pre-registered family.

**Rule-based composite and ML challenger.** The composite is a fixed-weight sum of four standardised
components (compression 0.30, delivery 0.25, positioning 0.25, volume 0.20). XGBoost (primary), random
forests and logistic regression use the same features and the same events. Hyper-parameters are chosen
by purged 5-fold cross-validation with a 10-day embargo on training and validation data; final models are
fitted on training data, early-stopped and isotonically calibrated on validation data. H4 compares
XGBoost with the composite on the test set by DeLong's test (DeLong et al., 1988; Sun and Xu, 2014) and
by top-decile precision. SHAP interaction values rank all feature pairs.

**Economic evaluation.** Signals are traded from the next open to the barrier exit, equal-weighted with at
most 20 concurrent positions, after Indian delivery costs (securities transaction tax, exchange and SEBI
fees, stamp duty, GST, brokerage, depository charges) and liquidity-dependent slippage. A benchmark ladder
asks whether each filter adds value over the previous one: base rate → all breakouts → VCP → composite ≥ 1
→ ML top decile. Strategy significance uses deflated Sharpe ratios over every logged variant and the SPA
and Reality Check tests against the Nifty 500.

## 5. Results

**5.1 Events and base rates.** After point-in-time exclusions (F&O ban, results days known at the time,
price below ₹20, short history), Tier 1 yields 4,876 labelled BROAD events (train 2,102, validation 978,
test 1,796), 11,327 twenty-day breakouts and only 82 STRICT (VCP) events (38 / 15 / 29). The ten-day
triple-barrier success rate is about 25% for every event family — below the 35–50% the design
anticipated. Every compared STRICT group has fewer than 30 events, so no STRICT hypothesis receives a
p-value (pre-registered rule); the power rule's widening to all liquid stocks (160 train+validation and
144 test STRICT events) also shows no effect (test: −0.2 points, p = 0.98).

**5.2 Hypotheses on training and validation data (exploratory).** Table 2 (`results/metrics/
p3_hypothesis_table.csv`) reports the six pre-registered tests. For BROAD events, success is 27.3%
with delivery expanding during the base and 26.2% without (+1.0 points, p = 0.51, BH q = 0.94); the
logistic model with size, volatility-regime, liquidity and sector controls gives a delivery coefficient
of 0.040 (p = 0.65) and a delivery × positioning interaction of 0.031 (p = 0.73). Divergence (H2:
26.7% vs 26.8%, p = 0.94) and the FII/DII regime (H3: asymmetry 1.03 vs 1.07, p = 0.80) are flat.

**5.3 Confirmatory tests on the test set (run once).** Table 3 (`results/metrics/
p6_confirmatory_table.csv`) gives the eight-test family. H1: 24.8% vs 24.9% (−0.04 points, p = 0.99).
H2: divergent breakouts succeed 23.5% of the time against 26.3% for the rest (−2.8 points, the
predicted direction, p = 0.17, q = 0.29). H3: asymmetry 1.01 vs 0.88 (p = 0.38). H4 is a **loss**:
XGBoost's AUC is 0.460 against 0.526 for the fixed-weight composite (difference −0.066, SE 0.023,
DeLong p = 0.005, q = 0.024); top-decile precision is 22.4% (n = 165) against 31.0% for the composite's
top decile (n = 226; −8.5 points, p = 0.11), and the return spread is −0.26 points (95% CI −1.29 to
+0.64). Random forest (0.466) and logistic regression (0.468) do no better; isotonic calibration
improves the Brier score (0.250 → 0.195) but not the ranking. An expanding-window walk-forward with
monthly refits and frozen hyper-parameters reaches an out-of-sample AUC of 0.502. Excluding the
period our earlier pilot had explored (from 28 July 2025), H2 strengthens to −5.7 points (p = 0.026)
but does not survive the false-discovery correction (q ≈ 0.13), and the H4 loss is no longer
significant (−0.039, p = 0.14). No hypothesis is supported.

**5.4 Why the ML challenger fails.** With validation PR-AUC as the early-stopping criterion, XGBoost
stopped after 14 depth-2 trees: there was almost nothing to learn. Its few interactions involve
market-wide positioning series (e.g. DII stock-futures COT with client index-futures divergence);
all three pre-registered delivery pairs have zero interaction (Figure 5 and the SHAP files), so the
novelty criterion is not met. Market-wide features change slowly, which makes them easy to overfit in
time and helps explain why a model that leans on them ranks test events worse than chance.

**5.5 Benchmark ladder and economic evaluation.** On the test set (Figure 1), twenty-day breakouts
succeed less often than an average Tier-1 stock-day (24.9% vs 26.7%; −1.9 points, p = 0.005);
compression adds nothing (24.8%); the composite rule (score ≥ 1) selects only 29 events with 41.4%
success (+16.8 points over the excluded BROAD events, p = 0.038) — suggestive, but below the 30-event
threshold and not part of the confirmatory family; the ML top decile selects 165 events at 22.4%.
After costs (Figure 6), trading every breakout loses 26.8% a year (Sharpe −4.0, maximum drawdown
−68.6%, profit factor 0.59) and BROAD breakouts lose 12.6% a year; the sparse composite and ML
strategies earn roughly the cash rate (6.0% and 4.3%), while the Nifty 500 returned 9.7% a year.
Hansen's SPA (p = 0.77) and White's Reality Check (p = 0.89) find no variant that beats the index, and
every deflated Sharpe ratio is zero after accounting for the 89 variants logged before the test run
(170 including the later robustness grid).

## 6. Robustness

**Sub-periods and COVID.** H1 for BROAD events is +1.0 points in 2015–2019 (p = 0.62) and −1.1
points in 2021–2022 (p = 0.74); within the test period success falls from 28.8% (2023) to 17.4% (2026
to date), with no consistent delivery effect in any year. Excluding March–May 2020 leaves every result
unchanged (H1 +0.9 points, p = 0.56).

**FII and DII separately.** Breakouts during FII selling alone or DII buying alone show no asymmetry
difference (all p > 0.96 on training and validation; p = 0.56 and 0.77 on the test set).

**Placebo.** Shuffling delivery percentages across stocks within each day (20 draws) gives H1
differences centred at −1.1 points (sd 1.25); the actual +1.05 points exceeds all 20 draws (Figure 8).
Delivery therefore carries some stock-specific information, but the effect on breakout success is
about one percentage point — economically negligible after costs.

**VCP parameters.** Across 81 settings of fractal width, maximum depth, volume multiple and base length,
STRICT counts range from 13 to 418 (Figure 7). Wider pivots (k = 5) produce 135–418 events and H1
differences between +4.8 and +15.8 points (mean +9.1); across the 72 settings with at least 30 events the H1 difference is
positive in 67% (mean +2.8 points). Because parameters may not be chosen on outcomes, we report this
surface only as a direction for future pre-registered work.

**Other checks.** A 104-week COT window does not change the composite (validation AUC 0.499 vs 0.506);
outside the F&O segment (Tier 2) delivery shows no effect (train+validation −0.3 points, p = 0.88;
test +0.8 points, p = 0.59).

**Exploratory lead-lag (not part of the pre-registered family).** Daily Nifty 500 returns strongly
Granger-precede changes in FII stock-futures positioning (p < 10⁻⁷ at lags 1–5) but not the reverse —
FIIs chase price in stock futures. Changes in FII *index*-futures positioning do precede index returns
(p = 0.001–0.02), and days when FII stock-futures positioning sits in the bottom decile of its year are
followed by higher 60-day index returns (+6.4% vs +2.9% unconditionally, overlapping windows). These
patterns are consistent with positioning acting as a contrarian market-level signal rather than a
stock-level breakout filter; they are supporting evidence only, not causal claims.

## 7. Limitations

- **Category coarseness.** NSE's "Client" category mixes retail traders and high-net-worth individuals;
  participant data are market-wide, so they act as a regime filter, not a stock-level signal.
- **Hedging ambiguity.** Institutional futures shorts are often hedges of cash holdings; without a
  historical daily cash-flow series we cannot separate hedging from directional positioning (the FII/DII
  "flow" is a futures-positioning proxy).
- **Two-tier asymmetry.** Positioning features exist only for F&O stocks; Tier-2 evidence covers delivery
  and compression only.
- **Structural breaks.** The test period spans T+1 settlement (2023) and the 2024–25 F&O reforms; results
  are reported by sub-period.
- **Contrarian vs continuation.** Positioning extremes are usually read as contrarian while the VCP is a
  continuation pattern; positioning is therefore used as a filter, not a trigger.
- **Data gaps.** No point-in-time market capitalisation, sector history or ASM/GSM lists; sizes are
  proxied by turnover and sectors are today's labels.
- **Multiple testing.** Many variants were tried; all are logged, and inference is corrected for them.
- **Prior exploration.** Part of the test period was explored by the authors' earlier pilot; results are
  reported with and without that window.

## 8. Conclusion

Using eleven years of official NSE data, a pre-registered design and a test set touched once, we find
no evidence that delivery behaviour, stock-level positioning or market-wide institutional flows —
alone or in interaction — improve the success of breakouts from compressed bases. A machine-learning
model given all 80 features does worse than a simple fixed-weight rule, and breakout trading loses
money after Indian transaction costs. The specified volatility contraction pattern is too rare in F&O
stocks to test. Three results deserve pre-registered follow-up: divergence between price and futures
open interest (H2) pointed the predicted way on the test set and strengthened outside the pilot
window; the composite rule's few high-score events succeeded far more often than the rest; and FII
index-futures positioning showed contrarian lead-lag properties at the market level. We release the
data pipeline — reconstruction of pre-2019 delivery data, verified point-in-time universes,
corporate-action adjustment and a no-look-ahead test suite — so that these questions can be studied
with the same safeguards.

## References

*(Verify every entry against the source before submission.)*

- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proceedings of KDD 2016*, 785–794.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837–845.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365–380.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.
- Lo, A. W., Mamaysky, H., & Wang, J. (2000). Foundations of technical analysis: computational algorithms, statistical inference, and empirical implementation. *Journal of Finance*, 55(4), 1705–1765.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
- Minervini, M. (2013). *Trade Like a Stock Market Wizard*. McGraw-Hill.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of ICML 2005*, 625–632.
- Sanders, D. R., Boris, K., & Manfredo, M. (2004). Hedgers, funds, and small speculators in the energy futures markets: an analysis of the CFTC's Commitments of Traders reports. *Energy Economics*, 26(3), 425–445.
- Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. *IEEE Signal Processing Letters*, 21(11), 1389–1393.
- Wang, C. (2003). The behavior and performance of major types of futures traders. *Journal of Futures Markets*, 23(1), 1–31.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.
