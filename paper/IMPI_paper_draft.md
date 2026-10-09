# Delivery, Positioning and Compression: Do Their Interactions Predict Breakout Success on the National Stock Exchange of India?

*Draft for a conference submission — Team 23 (Fazil T, Faizur Rahman S, Suhail Ahmed S), Department of
Computer Science and Engineering (AIML), Hindustan Institute of Technology & Science. Supervisor:
Dr. Jeyavim Sherin RC.*

> Status: Sections 1–4 and 7 are written; Sections 5–6, the abstract and the conclusion are generated
> from `results/` after the one-time test evaluation (see `results/summary.md`). Items marked
> **[TEAM]** need the team's input (literature on Indian markets, venue formatting).

## Abstract

{ABSTRACT}

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

{RESULTS}

## 6. Robustness

{ROBUSTNESS}

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

{CONCLUSION}

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
