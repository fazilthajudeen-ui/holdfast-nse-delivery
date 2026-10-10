"""Insert the results-dependent sections into paper/IMPI_paper_draft.md (numbers from results/metrics/*.json,
transcribed 10 Oct 2026 after the one-time P6 run; see results/summary.md for the generated tables)."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "paper" / "IMPI_paper_draft.md"
s = p.read_text(encoding="utf-8")

ABSTRACT = """We test whether the interaction of delivery behaviour, positioning and price compression predicts
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
pipeline, as a reproducible baseline for delivery-based research in India."""

RESULTS = """**5.1 Events and base rates.** After point-in-time exclusions (F&O ban, results days known at the time,
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
(170 including the later robustness grid)."""

ROBUSTNESS = """**Sub-periods and COVID.** H1 for BROAD events is +1.0 points in 2015–2019 (p = 0.62) and −1.1
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
stock-level breakout filter; they are supporting evidence only, not causal claims."""

CONCLUSION = """Using eleven years of official NSE data, a pre-registered design and a test set touched once, we find
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
with the same safeguards."""

for k, v in [("{ABSTRACT}", ABSTRACT), ("{RESULTS}", RESULTS), ("{ROBUSTNESS}", ROBUSTNESS), ("{CONCLUSION}", CONCLUSION)]:
    if k in s:
        s = s.replace(k, v)
s = s.replace("""> Status: Sections 1–4 and 7 are written; Sections 5–6, the abstract and the conclusion are generated
> from `results/` after the one-time test evaluation (see `results/summary.md`). Items marked
> **[TEAM]** need the team's input (literature on Indian markets, venue formatting).""",
              """> Status: complete first draft (10 Oct 2026). All numbers come from `results/metrics/*.json` and
> `results/summary.md`; figures are in `results/figures/`. Items marked **[TEAM]** need the team's input
> (literature on Indian markets, venue formatting, final citation checks).""")
p.write_text(s, encoding="utf-8")
print("remaining placeholders:", [x for x in ["{ABSTRACT}", "{RESULTS}", "{ROBUSTNESS}", "{CONCLUSION}"] if x in s])
