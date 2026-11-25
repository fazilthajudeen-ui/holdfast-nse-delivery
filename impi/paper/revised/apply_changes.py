# Builds main_revised.tex from baseline.tex (= main_single.tex without the Acknowledgment section,
# matching the project PDF). Each block replaces exactly one original passage, from START to END inclusive.
s = open("baseline.tex", encoding="utf-8").read()


def rep(start, end, new):
    global s
    assert s.count(start) == 1, ("start not unique/missing", start)
    i = s.index(start)
    j = s.index(end, i) + len(end)
    s = s[:i] + new + s[j:]


# 0. monospace: XeTeX would otherwise use Latin Modern Mono; the original (pdfTeX) used Courier (Nimbus Mono).
s = s.replace("\\begin{document}", "\\ifdefined\\XeTeXrevision\\setmonofont{texgyrecursor-regular.otf}\\fi\n\\begin{document}", 1)

# 0b. Author affiliations: Dept. of CSE (AI & ML) -> Dept. of DISC (first three authors)
assert s.count(r"Dept.\ of CSE (AI \& ML)") == 3
s = s.replace(r"Dept.\ of CSE (AI \& ML)", r"Dept.\ of DISC")

# 1. Abstract
rep(r"Indian traders read two free daily footprints", r"after costs no strategy beats the Nifty~500 index.",
r"""Two ``free'' daily tracks that traders in India watch to gauge informed activity in a stock are (1) the
delivery percentage of quantity traded that is settled into investors' accounts and (2) changes in futures
positions. Can these identify successful price breakout events? We recreate a survivorship-free,
corporate-action-adjusted panel from official National Stock Exchange of India (NSE) files (2014-01-01 to
2026-10-09, 3,748 symbols), mark breakouts by triple barriers (2 up average true range, 1 down; 10 days) and
assess four hypotheses that are all in version control before we conduct any confirmatory study. We check the
test set (January 2023 to 2026-10-09) once. No hypotheses are verified. Expanding deliveries affect success in
the test set by \ensuremath{-}0.04 percentage points (pp) ($p=0.987$). Price up, open interest (OI) down is the
right sign (\ensuremath{-}2.81 pp), but does not pass false discovery control ($q=0.286$). A gradient-boosted
model sorts breakouts worse than fixed-weight composite (0.460 vs.\ 0.526 area under the curve). The setup has
power to spot effects of size around 6.26 pp. So lower effects cannot be excluded. 95\% confidence intervals for
the delivery and OI effects in post-hoc analyses of the entire test set are inside the 8.8 pp swing in success
rate needed to pay for one round trip of Indian trading costs. No strategy's post-costs return exceeds the
Nifty~500 index.""")

# 2. Introduction, paragraph 1
rep(r"Theories of informed trading predict", r"supportive positioning is ``real''.",
r"""Recent IEEE studies forecast Indian and global stock prices with boosted trees, deep ensembles and
explainable AI \cite{samantaray2026,singh2025,kumar2025,rajagopal2026}. Intraday data have been used to construct indicators of informed trading and ``flow toxicity''
\cite{easley1996,easley2012}. Informed trading theories suggest that more informed traders imprint their
presence on price and order flow \cite{kyle1985,glosten1985}. These theories are only motivation; we don't
measure order flow toxicity. We don't use intraday data. We work with free end-of-day data of National Stock
Exchange of India (NSE). There are two footprints of informed activity that traders can read: 1) delivery
percentage, 2) positioning on futures (i.e., a stock's own open interest (OI), NSE reported market-wide positions
for foreign institutional investors (FIIs), domestic institutional investors (DIIs), proprietary desks, and
clients). It's believed that high delivery with supportive positioning on a breakout is ``real''.""")

# 3. Introduction, paragraphs 2-3 (hypotheses + contributions)
rep(r"Because published signals are often chosen", r"how far the null extends and where it does not.",
r"""Since published signals are often picked from a large set of trial specifications
\cite{harvey2016,bailey2014pseudo}, we designed the study to credibly provide a negative result
\cite{nosek2018}. Pre-registering 4 hypotheses on BROAD compression breakouts (Section~\ref{sec:method}):
\textbf{H1}: breakouts with a delivery expansion base tend to succeed more often. \textbf{H2}: breakouts with
price increasing while futures OI of the stock decreasing tend to fail more often. \textbf{H3}: breakouts when
FII net selling / DII net buying tends to have better gain/loss ratio. \textbf{H4}: Gradient-boosted tree model
(XGBoost \cite{chen2016}) with the same features is superior to a constant-weight composite in both area under
receiver operating characteristic curve (AUC) and top-decile precision.

Contributions: (i) IMPI: a pipeline to reconstruct a survivorship-free point-in-time (PIT) NSE panel including
pre-2019 delivery data that is not available via the current NSE interface. (ii) Pre-registered study design.
One time test set open. Publicly available timeline. (iii) A confirmatory null result with minimum detectable
effect, and a cost-based interpretation of relevant effect sizes. (iv) Post hoc robustness tests clearly
distinguished, to indicate the range and limits of the null.""")

# 4. Related work
rep(r"\emph{Informed-flow theory.} Kyle's", r"White's Reality Check \cite{white2000}, and log every trial.",
r"""\emph{Informed-flow theory.} Gradual insider trading of Kyle to assimilate information \cite{kyle1985}.
Glosten--Milgrom, adverse selection embedded in spread pricing \cite{glosten1985}. PIN (probability of informed
trading) and volume-synchronized PIN (VPIN) for measuring informed trading through trade imbalance
\cite{easley1996,easley2012}. Require trade-level data. We merely adopt the idea of daily footprint of informed
trading.
\emph{Institutional flows in India.} Fast selling of Indian stocks post unexpected tax-policy shock by FII and
historical return chasing \cite{thapa2026}. We cannot access daily flow (at the firm level) of FII. We can use
participant positioning as a market regime here. We have not seen peer-reviewed test of NSE delivery and/or
futures position as predictor of breakout success.
\emph{Machine learning (ML) for stock prediction.} Non-linear models are said to outperform linear (but not
often out-of-firm \cite{lu2024})
\cite{fieberg2023,azevedo2023,farimani2024}. On NSE data, Samantaray et al.\ tune a light gradient boosting
machine (LightGBM) with constricted particle swarm optimization on wavelet bands and add Dow Jones signals to
forecast three NSE indices \cite{samantaray2026}. Wavelet-denoised LSTM--CNN--TCN ensembles \cite{singh2025}
and a stacked XGBoost, LightGBM and SVR model \cite{kumar2025} also report low price errors on the Nifty~50.
These studies forecast index prices; we ask a different question, whether a single breakout succeeds. Now
widespread to use Shapley additive explanations (SHAP) for tree-ensembles \cite{lundberg2020} in ML for finance
\cite{cerneviciene2024,martins2024,yeo2025}. SHAP scores, together with out-of-sample Sharpe ratio and drawdown,
have also been used to screen machine-learned alpha factors on S\&P~500 stocks \cite{rajagopal2026}. Test
XGBoost versus simple rule.
\emph{Multiple testing and backtest overfitting.} Number of test $\rightarrow$ significance threshold should
increase \cite{harvey2016}. Deflated Sharpe ratio (DSR) adjusts for trials and nonnormality
\cite{bailey2014dsr}. We adopt Benjamini--Hochberg (BH) \cite{benjamini1995}. Hansen's test for superior
predictive ability (SPA) \cite{hansen2005}. White's reality check \cite{white2000}. Log each trial.""")

# 5. Data
rep(r"\emph{Data} (Fig.~\ref{fig:pipeline}; Table~\ref{tab:dataset}).", r"checked by automated quality control (QC).",
r"""\emph{Data} (Fig.~\ref{fig:pipeline}, Table~\ref{tab:dataset}). NSE data end-of-day data files have 3,161
trading days with 5.84 million daily rows of equities for 3,748 symbols (including delisted stocks). The F\&O
eligible symbols are selected PIT (354 underlyings, 134--227 daily). For delivery before 2019, the delivery file
is reconstructed from legacy equity file and security level delivery position file (MTO), which matches the
modern file (0 mismatches on 12 overlap dates). 1,442 split / bonus / demerger price, volume and OI adjustments
are made. QC of files is automated.""")

# 6. Universe and look-ahead
rep(r"\emph{Universe and look-ahead.}", r"truncated copies changed no past value.",
r"""\emph{Universe/Lookahead:} Tier~1: F\&O stocks (on that day, except for ban-list days) excluding result
announcement periods, prices less than Rs~20 and less history. Split: Train (June 2015--2020) Val (2021--2022),
Test (Jan 2023 to 2026-10-09). Purge labels crossing split boundaries. Features on day $T$ use data only until
23:59 Indian Standard Time (IST) on $T$. Entries at $T+1$ Open. No historical change when recomputing features
on 57 truncated copies.""")

# 7. Events and label
rep(r"\emph{Events and label.} STRICT events are", r"BROAD events with incomplete windows are excluded.",
r"""\emph{Events and label.} STRICT events are an algorithmic pattern of vol contraction (VCP)
\cite{minervini2013}. BROAD events are closes above previous 20 days high after compression (bollinger-width
percentile $<$30 for 5 days out of last 10); B1 = all closes at 20-day highs (Table~\ref{tab:events}). At entry
$E$ at $T+1$ open, $A$ = 14 day average true range (ATR), $y=1$ if $E+2A$ was reached before $E-A$ within 10
days, else $y=0$ \cite{lopezdeprado2018}. Day touching both barriers is a loss. 4 BROAD events with imperfect
windows are excluded.""")

# 8. Tests and models (before the equation)
rep(r"\emph{Tests and models.}", r"mean adverse excursion. The logistic model",
r"""\emph{Tests/Models.} 80 frozen features. Delivery, volume, illiquidity, compression, VCP, stock OI, COT
(participant positioning) style. Index and stock futures. Basis. Put/call ratio. Regime. Comparison of success
rates. H1, H2 (two-proportion $z$-test; block bootstrap, calendar month). H3: gain/loss ratio. Mean favourable /
mean adverse excursion. Logistic model.""")

# 9. After the equation
rep(r"where $D$ is the exposure and $P$ a second factor", "slippage per\nside, against the Nifty~500.",
r"""where $D$ is the exposure and $P$ is another variable (H1: alignment of positioning, H2: delivery
$z$-score), to estimate $\beta_3$ of the interaction, with controls for turnover, volatility, liquidity, sector,
$x$. Family of eight pre-registered tests (H1--H3 for STRICT, BROAD, H4-AUC, H4-top-decile) corrected by BH for
$q=0.10$. Group with less than 30 events doesn't have $p$-value. Isotonically calibrated to validation data:
XGBoost (random forest and logistic regression too) with purged, embargoed 5-fold cross validation tuning.
Comparison with composite by DeLong's test \cite{delong1988}. Trade Rs~10~lakh, at most 20 equal weight
positions with all Indian delivery costs, plus 5--20bp of slippage per side, vs Nifty~500, backtest.""")

# 10. Pre-registration and integrity
rep(r"Hypotheses, events, test family and decision rules were committed", r"reported without that window.",
r"""Hypotheses, events, test family and decision rules: version \texttt{15103f1} committed at 08:54 IST 10
October 2026 with frozen features (Table~\ref{tab:timeline}). \emph{Lock:} The single test run began at 09:23
IST on commit \texttt{ea49def} (code and training/validation results) and created the lock file, first
committed \texttt{359bcc7} at 09:30. \emph{Trials:} 67 of the 89 rows in the results log until the test
evaluation occurred before the test run. The test run created 22. The DSR used all 89 (conservative).
\emph{Disclosure:} The entire period was used for data download and corporate-action corrections and QC. The
pre-registration lists for its power calculation event counts and total success rate by split (including test)
but not results by split by any hypothesis exposure. For our pilot study we considered July 2025 until October
2026. For confirmatory results we also show them without that window.""")

# 11. Table III note
rep(r"\raggedright 67 of the 89 results-log rows", r"but no result split by any hypothesis exposure.}",
r"""\raggedright 67 of the 89 results (rows from log upto test evaluation) were pretest written, the other 22 are test run specific rows. The first time the lockfile was commited was in 359bcc7. The full period was used for data download, adjust fixes, QC (pre pre-registration). Pre registration has e.g. counts for events in test set, and the overall test success rate (power table). But no result per hypothesis exposure.}""")

# 12. Exploratory results
rep(r"On training and validation data, BROAD breakouts", r"\ValAUCcomp{} for the composite.",
r"""Training \& Validation data: BROAD with Delivery Expansion: 27.3\% success rate (vs 26.2\%: +1.0 pp,
$p=0.513$). H2: 26.7\% (vs 26.8\%, $p=0.936$). H3 ratio: 1.03 (vs 1.07, $p=0.801$). XGBoost has a validation AUC
of 0.492 (early stopping: 14 trees, depth of 2). Random forest AUC: 0.487. Logistic model AUC: 0.528. Composite
AUC: 0.506.""")

# 13. Confirmatory results
rep(r"All STRICT groups have fewer than 30 events per arm, so their", r"\TopCompPrec\% ($p=\TopP$).",
r"""All STRICT groups have less than 30 events per arm. Thus, $p$ values are suppressed and BH runs over $m=5$
finite $p$ values (Table~\ref{tab:confirm}). None's supported. \textbf{H1:} 24.8\% vs.\ 24.9\%
(\ensuremath{-}0.04 pp, $p=0.987$). \textbf{H2:} 23.5\% vs.\ 26.3\% (\ensuremath{-}2.81 pp, $q=0.286$);
omitting pilot window \ensuremath{-}5.65 pp ($p=0.026$, $q=0.128$). \textbf{H3:} 1.01 vs.\ 0.88 ($p=0.384$).
\textbf{H4:} XGBoost AUC 0.460 vs.\ 0.526 composite (difference \ensuremath{-}0.066, DeLong $p=0.005$,
$q=0.024$), only rejection (the wrong way); forest 0.466; logit 0.468; walk-forward (46 refits) 0.502; Top
decile precision 22.4\% vs.\ 31.0\% ($p=0.108$).""")

# 14. PSI / calibration paragraph
rep(r"The model leaned on market-wide positioning features", r"validation base rate (\BrierConstVal).",
r"""In particular, the model used widespread features of market positioning that changed their distributions.
We find a post hoc population stability index (PSI) of 0.55 for the put--call ratio and 0.31 for client
positioning in index futures, which exceeds the usual cutoff of 0.25. The calibration reduced the Brier score
\cite{brier1950} from 0.250 to 0.195, which is still worse than a constant forecast at validation base rate
(0.188).""")

# 15. Power
rep(r"\emph{Power.} The design has 80\%", r"pp, $p=\TostHTwoP$).",
r"""\emph{Power.} The power of the design is 80\% for $\alpha=0.05$ with 6.26 pp (H1), 5.71 pp (H2). Smaller
effects are possible. No equivalence is shown within $\pm$2 pp (post-hoc two one-sided tests (TOST)
\cite{lakens2017}: 90\% intervals [\ensuremath{-}3.7, +3.6] pp ($p=0.19$) and [\ensuremath{-}6.2, +0.6] pp
($p=0.65$)).""")

# 16. Table IV note
rep(r"\raggedright PRE-REGISTERED. Diff.\ in pp", r"No hypothesis is supported.}",
r"""\raggedright PRE-REGISTERED DIFFERENCE TESTS. We test for the difference in pp for H1, H2 and H4-top-decile, and difference of gain-to-loss asymmetry ratio for H3. For H4-AUC we test the difference of AUC between XGBoost and composite model. For STRICT tests, the number of events per arm is less than 30. In such case the pre-registered guard suppresses $p$-values (but shows unguarded $p$-values in brackets for informational purposes only, not included in analysis). We then use the Benjamini--Hochberg (BH) correction for 5 finite $p$-values ($m=5$). We only get significant rejection of H4-AUC (but the rejection direction is the wrong way, suggesting XGBoost is worse than composite model in ranking ability). So we get no support for any of the hypothesis.}""")

# 17. Backtests (footnote text kept verbatim)
fn = s[s.index(r"\footnote{Fewer trades"):]
fn = fn[:fn.index("signals).}") + len("signals).}")]
rep(r"After costs (Table~\ref{tab:backtests}) all breakouts", r"Fig.~\ref{fig:ladder}).",
r"""In Table~\ref{tab:backtests}, all the breakouts gave \ensuremath{-}26.8\% pa (BROAD \ensuremath{-}12.6\%)
after costs. Composite (29 trades) and ML top decile (157 trades)""" + fn + r""": +6.0\%, +4.3\%, vs.\
Nifty~500's +9.7\%. So this is close to the 6\%pa idle cash earns in this backtest. Also note the Sharpe's:
\ensuremath{-}0.01 and \ensuremath{-}0.85. So no outperformance over cash. No strategy outperforming benchmark
according to SPA ($p=0.77$) and Reality Check ($p=0.89$). Breakouts win less often than all stock-days that
could potentially be breakouts (24.9\% vs.\ 26.7\%, see Fig.~\ref{fig:ladder}).""")

# 18. Fig. 2 caption
rep(r"\caption{Pre-registered benchmark ladder", r"not plotted.}",
r"""\caption{Pre-registered benchmark ladder during test period. Success rate for 10-days with Wilson 95\%
intervals (dashed line is total eligible stock-days base rate). Rungs with $<$2 events are identified but not
graphed.}""")

# 19. Post-hoc lead sentence
rep(r"\emph{All analyses in this subsection", r"test set was opened.}",
r"""\emph{The tests in this subsection were all post-hoc analyses developed after opening the test set.}""")

# 20. Cost break-even
rep(r"\textbf{Cost break-even.}", r"excluding zero but not the break-even.",
r"""\textbf{Cost break-even.} Average round-trip costs are 0.58\% (range 0.45--0.75\%). If the gross profits on
successful and unsuccessful events are +4.74\% and \ensuremath{-}1.85\%, respectively, then a success-rate change
$d$ only pays for 1 round trip if $d\ge 8.8$ pp. 95\% intervals for H1 (\ensuremath{-}4.4 to +4.3 [month block
\ensuremath{-}5.3 to +4.7]), H1 without the pilot (\ensuremath{-}2.8 to +8.1), and H2 (\ensuremath{-}6.8 to
+1.2) are within $\pm$8.8 pp. H2 without pilot-window (\ensuremath{-}5.65 pp [\ensuremath{-}10.6 to
\ensuremath{-}0.7]) hits the break-even point. For the composite top decile +7.0 pp [+0.6 to +13.4] above other
BROAD events (not including zero, but including the break-even point).""")

# 21. Placebo
rep(r"\textbf{Placebo.} Shuffling delivery", r"placebo draws there, is not reproduced.",
r"""\textbf{Placebo.} The average placebo H1 effect is not 0, but +1.65 pp (SD 1.97), if you shuffle delivery
across stocks within each day 1,000 times. The delivery increase is a $z$-score to the trailing year per stock,
and since you shuffle within each day, the market-wide delivery level is constant, so there is a time-varying
market component in the placebo exposure. This isn't a clean null. The real \ensuremath{-}0.04 pp is at 20th
percentile (one-sided $p=0.795$, Fig.~\ref{fig:placebo}). There is no other stock-specific delivery
information, beyond this component. And the H1 effect in training/validation of +1.05 pp, which is higher than
all the 20 placebo draws in there, is not reproduced.""")

# 22. Fig. 3 caption
rep(r"\caption{Post-hoc placebo on the test set", r"(see text).}",
r"""\caption{Post-hoc placebo (test set). H1 effect for within-day shuffling of delivery among stocks. Real
effect shown. The shuffled distribution is non-zero-centered (see text).}""")

# 23. Specification curve
rep(r"\textbf{Specification curve.}", "The specifications overlap and are not\nindependent.",
r"""\textbf{Spec curve.} With varying barrier/horizon/threshold for delivery/event set (Fig.~\ref{fig:spec})
\cite{simonsohn2020}, 89\% of the 54 H1 specifications are in the predicted direction, 15 sig 5\% (0 in
opposite), 3 BH surviving. Median +2.36 pp, max 5.05 pp $<$ break-even. The preregistered specification
(\ensuremath{-}0.04 pp) is ranked 49 of 54. Higher thresholds, wider barriers lead to higher effects. H2: 3 sig
of 18, 0 surviving, median \ensuremath{-}2.24 pp. Specification frozen ranked 6. The specs are overlapping and
not independent.""")

# 24. Fig. 4 caption
rep(r"\caption{Post-hoc specification curve on the test set", r"the pre-registered specification.}",
r"""\caption{Test set post-hoc specification curve. Top: Effect and 95\% confidence interval for each
specification. Bottom: Choices for specification. Square shows the pre-registered specification.}""")

# 25. Bayes factors
rep(r"\textbf{Bayes factors.} BF$_{01}$", r"only under wide priors \cite{leewagenmakers2013}.",
r"""\textbf{Bayes factors:} 17.88 (uniform prior \cite{kass1995}) and 8.41 (unit-normal log odds prior
\cite{gronau2021}) for H1 (BF$_{01}$), only 2.27--2.32 with sceptical priors; for H2: 7.65, 3.63 and
1.05--1.13. If no pilot window, H2 under sceptical beta prior: 0.23 (moderate evidence for difference). Only
strong evidence for null if wide priors \cite{leewagenmakers2013}.""")

# 26. Cluster-robust model
rep(r"\textbf{Cluster-robust model.}", r"with no multiplicity adjustment.",
r"""\textbf{Cluster-robust model.} Two-way date $\times$ stock clustering \cite{cameron2011}. In
\eqref{eq:logit}, the test-set interaction is $\beta_3=+0.246$ [\ensuremath{-}0.130, +0.622] ($p=0.200$) for H1
and \ensuremath{-}0.186 ($p=0.142$) for H2. +0.185 ($p=0.043$) is a single post-hoc coefficient (not
multiplicity-adjusted) for the H2 model for delivery.""")

# 27. Other checks
rep(r"\textbf{Other checks.}", r"on wider next-day ranges.",
r"""\textbf{Other tests.} H1 is \ensuremath{-}1.4 pp ($p=0.572$) without 5 sessions before 47 F\&O expiries.
None of 12 yearly H1 estimates are significant. Same day barrier ties are 0.12\% of breakouts loss. More often
breakouts are stopped out day 1 with bigger next day ranges (15.0\% vs 10.3\%).""")

# 28. Discussion
rep(r"The pre-registered answer is no:", "that mostly fail multiple-testing control and stay below break-even.",
r"""PREREGISTERED: No effect on breakout success by delivery expansion, OI divergence or FII/DII regime, boosted
trees $<$ fixed composite, suggestive of drift in market-wide positioning features. NULL only informative on
large effects (Detectable effect $\sim$6.26 pp, not establishing $\pm$2 pp equiv., only strong null support from
Bayes factors with wide priors). Full test interval post hoc $<$ 8.8 pp cost break-even. 1 sub-sample doesn't
fit: without pilot window, H2 = \ensuremath{-}5.65 pp ($p=0.026$, $q=0.128$), 95\% interval touches
break-even. BF$_{01}$ (scept prior) 0.23 is moderate evidence for H2. This is a lead (post-hoc sub-sample
failing false discovery control). Pre-registered H1 spec among weakest (as in spec curve); maybe delivery
threshold stricter or wider barriers, post hoc effect a few pp (maybe information on move size but not
direction), mostly not multiply controlled, $<$ break-even.""")

# 29. Limitations
rep(r"\emph{Limitations.} Participant positions", r"chosen after the result was known.",
r"""\emph{Limitations:} Market-wide positions for participants only. Futures positions may be hedges.
``Client'' class includes both retail and wealthy. Too few STRICT events for inference. Data covers T+1
settlement (2023) and F\&O reforms (2024--2025). Full dataset preparation done prior to registration (with
revealed counts/base rates in test set), our pilot partially overlapped test period. Many post-hoc analyses.
Choices made post-knowledge of result.""")

# 30. Conclusion
rep(r"With a survivorship-free NSE panel, hypotheses fixed", r"Nothing here is investment advice.",
r"""We study hypothesis testing on a NSE panel free from survivorship, where test set is only opened once and
hypothesis is pre-registered. The results show no evidence supporting the popular notion that delivery \&
futures positioning footprints can tell you about breakout success. There is no strategy that outperforms
Nifty~500, after costs. Though we do not completely rule out smaller effect, these effect sizes are below
Indian cost requirements on the full test set. The following four leads require pre-registered tests with
post-October-2026 data: (1) is FII price follower in stock futures? (2) Does FII positioning in index futures
lead index returns? (3) Stricter delivery threshold for breakout size prediction? (4) does price up OI down
divergence (H2) predict breakout failure. (it may according to post-hoc subset except pilot window). Also, we
intend to build a market-level regime transition model based on participant positioning to predict volatility
regimes. This will be pre-registered and will only be tested on post October 2026 data. This is not investment
advice.""")

# 31. Data and code availability
rep(r"Inputs are public NSE end-of-day files", r"are available from the authors on request.",
r"""Input files are publicly available NSE end-of-day files which are re-downloaded by the pipeline. The authors
can provide code and configuration, locked and post-hoc results, sources for each table/figure/number
(generators), and SHA-256 checksums for locked files upon request.""")

# 32. References: IEEE Reference Guide details only (article numbers, publisher city/state/country).
for old, new in [
    (r"\emph{Appl. Intell.}, vol.~55, no.~7, p. 601,", r"\emph{Appl. Intell.}, vol.~55, no.~7, Art. no.~601,"),
    (r"\emph{Appl. Intell.}, vol.~55, no.~1, p.~21,", r"\emph{Appl. Intell.}, vol.~55, no.~1, Art. no.~21,"),
    (r"\emph{Artif. Intell. Rev.}, vol.~57, p. 216,", r"\emph{Artif. Intell. Rev.}, vol.~57, Art. no.~216,"),
    (r"\emph{Artif. Intell. Rev.}, vol.~58, no.~6, p. 189,", r"\emph{Artif. Intell. Rev.}, vol.~58, no.~6, Art. no.~189,"),
    (r"New York: McGraw-Hill, 2013.", r"New York, NY, USA: McGraw-Hill, 2013."),
    (r"Hoboken, NJ: Wiley, 2018.", r"Hoboken, NJ, USA: Wiley, 2018."),
    (r"Cambridge: Cambridge" + "\n  University Press, 2013.", r"Cambridge, U.K.: Cambridge" + "\n  Univ. Press, 2013."),
]:
    assert s.count(old) == 1, old
    s = s.replace(old, new)

# 33. New IEEE references (verified on Crossref; PDFs in ieee_papers/), drop two less relevant ML papers,
#     then renumber the bibliography in order of first citation (IEEE rule).
import re
new_items = {
"samantaray2026": r"""S.~P. Samantaray, S.~Mohapatra, and M.~Mishra, ``A wavelet-decomposed
  {$\Delta$}-band with constricted {PSO}-tuned light gradient boosting machine for {Indian} stock market
  prediction with cross-country market influence,'' \emph{IEEE Access}, vol.~14, pp. 87\,746--87\,773, 2026,
  doi: \url{10.1109/ACCESS.2026.3700317}.""",
"singh2025": r"""P.~Singh, M.~Jha, and H.~Patel, ``Wavelet-enhanced deep learning ensemble for accurate
  stock market forecasting: A case study of {Nifty} 50 index,'' \emph{IEEE Access}, vol.~13, pp.
  87\,036--87\,047, 2025, doi: \url{10.1109/ACCESS.2025.3568634}.""",
"kumar2025": r"""C.~R. Kumar and S.~Manikandan, ``{SEMP-TA}: A novel stock market prediction approach
  based on stacking ensemble machine learning for effective trend analysis,'' \emph{IEEE Access}, vol.~13, pp.
  117\,479--117\,490, 2025, doi: \url{10.1109/ACCESS.2025.3586233}.""",
"rajagopal2026": r"""S.~Rajagopal, H.~Natraj, M.~Pavani, R.~Kumar, and R.~R. Roy, ``Automated alpha factor
  discovery using machine learning, deep learning, and explainable {AI} for transparent trading strategies,''
  \emph{IEEE Access}, vol.~14, pp. 116\,342--116\,358, 2026, doi: \url{10.1109/ACCESS.2026.3716293}.""",
}
REMOVE = {"lu2025", "gonzaleznunez2025"}
head, rest = s.split("\n\\BIBdecl\n", 1)
bib, tail = rest.split(r"\end{thebibliography}", 1)
entries = {}
for chunk in re.split(r"(?=\\bibitem\{)", bib)[1:]:
    key = re.match(r"\\bibitem\{([^}]+)\}", chunk).group(1)
    entries[key] = chunk.strip()
for k in REMOVE:
    entries.pop(k)
for k, v in new_items.items():
    entries[k] = "\\bibitem{%s}\n%s" % (k, v)
body = head[:head.index(r"\begin{thebibliography}")]
order = []
for grp in re.findall(r"\\cite\{([^}]+)\}", body):
    for k in grp.split(","):
        k = k.strip()
        if k not in order:
            order.append(k)
assert set(order) == set(entries), (set(order) ^ set(entries))
s = head + "\\BIBdecl\n\n" + "\n\n".join(entries[k] for k in order) + "\n\n\\end{thebibliography}" + tail

open("main_revised.tex", "w", encoding="utf-8").write(s)
print("ok")
