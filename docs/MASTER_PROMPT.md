MASTER PROMPT — DELIVERY LENS + IMPI (complete project state as of 10 Oct 2026)
=================================================================================

ROLE
You are taking over a finished, working final-year project. Read this whole brief before answering.
Use it to explain, present, defend in a viva, maintain, extend, or write about the project.
Every number here is real and comes from the project's own result files. Never invent numbers or
citations. If a figure isn't here, compute it from the files or say it is unknown. Always say which
date a number is "as of", and whether a result is pre-registered, exploratory, or post-hoc.

-----------------------------------------------------------------------
1. IDENTITY
-----------------------------------------------------------------------
- Project: "Delivery-Based Movement Analysis on NSE" (product: Delivery Lens), extended into
  IMPI — Indian Market Positioning Intelligence (research system + institutional-style terminal).
- Folder: C:\Users\fazil\delivery-movement-nse  (Delivery Lens at the root; IMPI in impi\ with its own
  git repo and Python venv impi\.venv). Delivery Lens root is not a git repo.
- Nature: research + education. NOT investment advice. Never present scores as buy/sell signals.
- Target: a conference paper within ~3 months of 10 Oct 2026 (by ~Jan 2027). The team wrote the paper
  in a separate chat; a draft also exists at impi\paper\IMPI_paper_draft.md.

-----------------------------------------------------------------------
2. PART A — DELIVERY LENS (the original project, still running nightly)
-----------------------------------------------------------------------
Idea: NSE publishes each stock's DELIVERY % every evening (share of traded shares taken into demat).
Question: do sharp rises (>=20% in 20 days) on unusually high delivery hold their gains better?
- 1-year pilot on Nifty 500 (data from 28 Jul 2025). Conviction Score = percentile (0-100, shown capped at
  99) of the 5-day average gap between delivery % and the stock's own previous-60-day median, ranked
  within its last 250 days. <30 Low, 30-70 Neutral, >=70 High. Buyer zones = 10-day EMA of the gap,
  enter >=+5 pts, exit <+2 pts (descriptive only).
- Results as of 8 Oct 2026 (rebuild nightly): 679 sharp rises in 360 stocks; High-Low excess vs Nifty
  500 = +2.4 pts at 20 days (p = 0.04, significant), +1.7 pts at 60 days (p = 0.49, not significant);
  5-day gap no longer significant (+0.7, p = 0.26). Honest line: holds better over weeks; 3 months not proven.
- Built: pipeline src\ (download, clean, features, events, stats, score, regime, report, update_daily),
  FastAPI website (landing, about, TradingView-style terminal with Lightweight Charts v5, watchlist, gauge,
  zones, drawing tools, rule-based "Ask Lens" assistant), Streamlit backup, 6-slide deck (teal/amber).
- Daily: Windows Task Scheduler "NSE Delivery Update", Mon-Fri 19:30 -> scripts\run_daily.bat.
- On 10 Oct the 9 Oct run had failed (process killed, code 0xC000013A); it was re-run: data now to 9 Oct 2026.
- NSE data issue found later: sec_bhavdata_full for 11 Sep 2026 has DELIV_PER inconsistent with
  DELIV_QTY/volume for 98% of stocks (affects Delivery Lens on that date; IMPI treats it as missing).

-----------------------------------------------------------------------
3. PART B — IMPI RESEARCH (pre-registered study, finished)
-----------------------------------------------------------------------
Research question: does the INTERACTION of positioning changes, delivery behaviour and price
compression predict (a) breakout success and (b) volatility expansion in Indian equities, beyond each
signal alone? Built from a detailed master spec (guardrails G1-G7, phases P0-P8).

KEY DECISIONS TAKEN WITH THE USER (10 Oct 2026)
- NSE's participant-wise OI (FII/DII/Pro/Client) exists ONLY as market-wide totals (5 rows per day), never
  per stock. So positioning = stock-level futures/options OI from the F&O bhavcopy + participant data as a
  market-wide regime/filter. H2/H3 reworded accordingly before registration.
- IMPI built as a new package beside Delivery Lens (untouched). Conference target ~3 months. Institutional
  UI after the research. User then asked to "finish in one go": human gates became code guards.
- Earlier, the user asked to put "achievable" numbers in the PPT; this was declined — real results only.

DATA (official NSE archives only, 1 Jan 2014 - 9 Oct 2026, frozen before analysis)
- 3,161 trading days; ~5.83M daily equity rows; 3,748 symbols incl. delisted; point-in-time F&O list from
  each day's F&O bhavcopy (354 underlyings over time, 136-216 per day).
- sec_bhavdata_full valid only from 1 Oct 2019 (NSE served a 27 Jun 2019 file for 30 Sep 2019 — rejected).
  Before that: legacy equity bhavcopy + MTO delivery file, matching sec_bhavdata_full EXACTLY (0 mismatches)
  on 12 overlap dates. F&O legacy (to 5 Jul 2024) vs UDiFF (from 8 Jul 2024): exact match on 5 overlap days.
- QC: 92 stale holiday files rejected; missed-session detector found 2 weekend sessions (22 Mar 2014,
  28 Feb 2015 Budget) which were then downloaded; 0 unresolved weekdays; India VIX for 2014 from NSE's
  historicalOR/vixhistory API.
- 1,442 split/bonus/demerger adjustments of prices AND quantities (840 NSE-confirmed by the price gap,
  468 gap-detected splits/bonuses missing from NSE's API, 133 demergers); unexplained gaps not adjusted.
  Fixes found on real data: abbreviated subjects ("Fv Splt Frm Rs 10 To Rs 2", ICIL), stale ex-date opens
  in old bhavcopies (ADANIENT 2015 demerger: open 0.90x, real 0.17x), downward-only gap snaps.
- Gaps that cannot be filled: no per-stock participant positions, no historical daily FII/DII cash flows
  (FII/DII "flow" = 20-day change in futures net-long share), no historical ASM/GSM lists, no point-in-time
  market cap (turnover used) or sector history (today's labels).

DESIGN
- Splits (purged): train 2015-06-01..2020-12-31, validation 2021..2022, test 2023-01-01..2026-10-09.
- Tier 1 = F&O stocks that day; Tier 2 = EQ stocks with 60-day median turnover > Rs 5 cr.
- Exclusions: F&O ban T/T+1, results day +/-3 (only if announced by T), price < Rs 20, < 30 listing days,
  < 252 days history.
- No look-ahead (G1): release times per dataset; decision T 23:59, entry T+1 open; truncation test on real
  data passed (57 checks, max difference 0).
- Events: STRICT = algorithmic VCP (fractal k=2 pivots, 2-4 contractions in a 5-15 week base, strictly
  declining depth and volume, close > pivot x1.001 on volume >= 1.5x average); BROAD = close above prior
  20-day high after compression (BB-width percentile < 30 on >= 5 of 10 days); B1 = all 20-day breakouts.
- Label: triple barrier — entry T+1 open, +2 ATR14 / -1 ATR14, 10 days (20/60 secondary); same-bar double
  touch = loss. Secondary: volatility expansion RV10 after / before >= 1.5.
- 80 frozen features (delivery z-score, volume, Amihud, compression, VCP, stock OI/COT/basis/PCR,
  FII/DII/Pro/Client COT in index and stock futures, regime).
- Stats: two-proportion z, calendar-month block bootstrap, logistic with controls + delivery x positioning
  interaction (month-clustered SE), BH FDR q=0.10, DeLong, deflated Sharpe, Hansen SPA, White Reality Check,
  n<30 rule. Costs: full Indian delivery costs (STT 0.1% each side, etc., Rs 20 brokerage, DP charge,
  5/10/20 bps slippage), Rs 10 lakh, max 20 positions, idle cash earns 6%.
- Integrity: hypotheses committed to git BEFORE any test (commit 15103f1, 10 Oct 2026 08:54 IST); test set
  evaluated ONCE (lock file, commit ea49def, 09:23-09:29). The Delivery Lens pilot explored Jul 2025-Oct 2026
  (inside the test period), so results are also reported without that window.

EVENTS (Tier 1, eligible; train / validation / test)
BROAD 2,102 / 978 / 1,796 (4,876); all breakouts 5,075 / 1,979 / 4,273 (11,327); STRICT 38 / 15 / 29 (82).
10-day success base rate ~25% (spec guessed 35-50%). VCP is rare for every pivot rule (D-041), kept as
specified; power rule widened STRICT to Tier 1 or 2 (160 train+val, 144 test events) — no effect.

RESULTS — TRAIN + VALIDATION (exploratory)
BROAD H1 27.3% vs 26.2% (+1.0 pt, p=0.51, q=0.94); logit delivery coef 0.040 (p=0.65), interaction 0.031
(p=0.73). H2 26.7% vs 26.8% (p=0.94). H3 asymmetry 1.03 vs 1.07 (p=0.80). STRICT: insufficient data.
Composite AUC (BROAD): train 0.487, validation 0.506. v1 ML validation AUC: XGB 0.492, RF 0.487, logit 0.528;
XGBoost early-stopped at 14 depth-2 trees; SHAP: all 3 pre-registered delivery pairs had zero interaction.

RESULTS — TEST SET (confirmatory, run once) — NO HYPOTHESIS SUPPORTED
H1 BROAD 24.8% vs 24.9% (-0.04 pt, p=0.99). H2 23.5% vs 26.3% (-2.8 pt, predicted direction, p=0.17, q=0.29).
H3 1.01 vs 0.88 (p=0.38). H4 = LOSS: XGBoost AUC 0.460 vs composite 0.526 (diff -0.066, SE 0.023, DeLong
p=0.005, q=0.024); top-decile precision 22.4% (n=165) vs 31.0% (n=226), p=0.11. RF 0.466, logit 0.468;
walk-forward (46 monthly refits) 0.502. Without pilot window: H2 -5.7 pt (p=0.026, q=0.128, not supported);
H4 -0.039 (p=0.14). Tier 2 H1 +0.8 pt (p=0.59).
Ladder (test): base 26.7% (159,972 stock-days); breakouts 24.9% (-1.9 pt vs non-breakout days, p=0.005);
BROAD 24.8%; composite >= 1: 41.4% but n=29 (exploratory); ML top decile 22.4% (n=165).
Backtests (test, after costs): all breakouts CAGR -26.8%, Sharpe -4.02, max DD -68.6%; BROAD -12.6%,
Sharpe -3.25, max DD -39.2%; composite 6.0% (29 trades); ML top decile 4.3% (157 trades); Nifty 500 +9.7%.
SPA p=0.77, Reality Check p=0.89; all deflated Sharpe = 0 (89 trials logged before the test).

ROBUSTNESS (train + validation)
Sub-periods: H1 +1.0 pt 2015-19 (p=0.62), -1.1 pt 2021-22 (p=0.74); test-year success 28.8% (2023) ->
17.4% (2026 YTD). COVID months excluded: unchanged. FII-only / DII-only H3: nothing. Placebo (delivery
shuffled within day, 20 draws): placebo mean -1.08 pt (sd 1.25); actual +1.05 pt beats all 20 -> delivery
carries some information but the effect (~1 pt) is economically negligible. VCP grid (81 settings): 13-418
events; k=5 gives 135-418 events with H1 +4.8 to +15.8 pt (exploratory only). 104-week COT: no change.
Exploratory lead-lag: Nifty 500 returns Granger-precede FII stock-futures positioning (p<1e-7) — FIIs chase
price; FII index-futures positioning precedes index returns (p=0.001-0.02); FII stock-futures COT in bottom
decile -> next 60-day Nifty 500 +6.4% vs +2.9%.

HONEST CONCLUSION: no evidence that delivery, positioning or institutional flows (alone or in interaction)
improve breakout success; ML did worse than a simple rule; breakout trading loses after Indian costs; the
specified VCP is too rare to test. Leads for future PRE-REGISTERED work on new data: H2 price-OI divergence,
the composite's high-score events, FII index-futures positioning as a contrarian market signal.

-----------------------------------------------------------------------
4. PART C — IMPI MODEL v2 (post-hoc, after the one-time test) + LIVE SYSTEM
-----------------------------------------------------------------------
The user asked to maximise model performance. Because the test set was already used, v2 was developed on
train+validation ONLY and its 2023+ numbers are always labelled POST-HOC.
- Training data: every Tier-1 stock-day (285k rows, vs 2,102 events in v1); 17 extra features (momentum
  1/5/60/120d, distance from 52-week high/low, realised-vol term structure and percentile, 5-day delivery
  z, delivery trend/rank, turnover shock, close location, gap, up-days); cross-sectional rank transforms.
- Targets: breakout success (label_10), beats the day's median F&O stock (label_xs10), volatility expansion.
- Selection: walk-forward CV, yearly folds 2018-2022, purged, 15-day embargo; feature sets x XGBoost configs
  x logistic x probability-average ensemble; learning-to-rank (pairwise, by date) also tried and NOT adopted
  (best CV 0.517 < 0.525). 198 rows in results_log.csv overall.
- Results (CV AUC -> post-hoc test AUC):
  * Breakout success: ensemble(XGB+logit), v1 features: 0.538 -> 0.505 (no edge; calibrated Brier 0.201 vs
    base-rate 0.196; BROAD events AUC 0.502).
  * Beats median F&O stock: ensemble: 0.525 -> 0.524 (small, stable edge; daily rank IC 0.049; top decile beats
    the average by +0.18% per 10 days gross, t=1.43, -0.27% after costs -> not tradable).
  * Volatility expansion: XGBoost, v1+v2 features: 0.742 -> 0.743 (useful; Brier 0.144 vs 0.163; event rate
    rises from 2% in the lowest score decile to 50% in the highest).
- Live mode: `python -m impi.live` (run nightly after Delivery Lens via scripts\run_daily.bat) downloads new
  days, rebuilds into impi\data\live (research folders never touched), scores all F&O stocks with the frozen
  v2 models, exports files for the web app, and appends predictions to impi\data\track\predictions.csv
  BEFORE outcomes exist (first logged 9 Oct 2026, 213 stocks). Matured outcomes are scored in summary.json
  after 10 trading days — the only truly out-of-sample test left. Files for the last 3 days are never
  recorded as "no data" (NSE may publish after 19:30), so they are retried.

-----------------------------------------------------------------------
5. PART D — THE TERMINAL (institutional-style software)
-----------------------------------------------------------------------
Run: from C:\Users\fazil\delivery-movement-nse -> `python -m app.server` -> http://localhost:8000
- /terminal: existing Delivery Lens terminal + new IMPI MODEL panel for F&O stocks (breakout / beats-median /
  volatility probabilities vs base rates with the day's percentile, delivery-compression-positioning snapshot,
  top XGBoost drivers per target, model-quality note) and IMPI breakout markers on the chart (toggle "IMPI":
  circle = compression breakout, square = VCP). Non-F&O stocks show a clear "not covered" note.
  Hash navigation / back button load stocks; top-bar links Scanner / Market / Research.
- /scanner: all F&O stocks today, sortable; filters by compression / VCP / 20-day breakout and sector.
- /market: FII/DII/Pro/Client futures positioning (net-long share and 52-week range, index vs stock futures),
  Nifty 500, India VIX, regime tiles (trend, VIX percentile, breadth, FII/DII 20-day flow).
- /research: pre-registered results, v2 model quality (CV + post-hoc + deciles), live track record, backtests,
  all figures.
- API: /api/impi/meta, /scanner, /stock/{symbol}, /market, /research (app\impi_api.py reads exported files;
  thread-safe reload; 503 with a clear message until the live export exists).
- Ask Lens (rule-based, NO external AI/LLM): new intents "IMPI model view", "Today's breakouts",
  "Market positioning", "Does the IMPI model work?" (honest answers).
- Light/dark themes, desktop + phone layouts verified in the browser; validated chart palette.
- Isotonic probabilities are step functions (breakout prob is ~25.9% for most stocks — the model separates
  weakly), so percentiles are shown next to probabilities (D-046).

-----------------------------------------------------------------------
6. TECH, QUALITY, ENVIRONMENT
-----------------------------------------------------------------------
- Python 3.12; IMPI venv: pandas 3.0.6, numpy 2.2.6, pyarrow 26, scikit-learn 1.9.1, xgboost 3.4.1,
  shap 0.53, statsmodels 0.15, arch 8.0 (SPA), scipy, matplotlib; Delivery Lens: FastAPI, pandas, Plotly.
- Tests: 44 IMPI tests (parsers, adjustment, no-look-ahead truncation, VCP, triple barrier, purged CV,
  DeLong vs bootstrap, BH vs statsmodels, cost model, backtest accounting, composite, v2 row-wise scoring,
  live guard) + 38 Delivery Lens tests (incl. 4 IMPI API tests). Full synthetic end-to-end rehearsal of
  P1-P8 (tests\integration\make_synthetic.py) before the real run.
- Laptop: Windows 11, 7.3 GB RAM (code is memory-light, per-year/per-stock). Windows Smart App Control blocks
  pyarrow.dataset and statsmodels.api — code imports narrowly (pandas filtering, GLM directly). Never change
  security settings.
- NSE downloads: polite (>=1 s, cookies, retries, session rebuild after stale connections), cached, resumable;
  ~17,000 files for 2014-2026. Per-day caches (F&O aggregates, parsed equity) make rebuilds fast.

-----------------------------------------------------------------------
7. FILES AND COMMANDS
-----------------------------------------------------------------------
impi\
  config\*.yaml (all numbers) · data\release_lag.yaml · pre_registered_hypotheses.md · decisions.md (D-001..D-046)
  README.md · paper\IMPI_paper_draft.md · results\summary.md · results\metrics\*.json · results\figures\fig1-8
  results\results_log.csv · results\test_lock.json · data\track\ (forward predictions)
  src\impi\ ingest transforms features labels research models backtest validation pipeline v2 live.py
app\ server.py impi_api.py assistant.py static\ (terminal, scanner, market, research, impi.css, impi.js)
scripts\run_daily.bat (Delivery Lens update, then IMPI live update)
Commands (impi, with impi\.venv):
  .venv\Scripts\python -m pytest
  .venv\Scripts\python -m impi.pipeline.run_all --to p2      # research pipeline (P6 is never chained)
  .venv\Scripts\python -m impi.v2.panel ; -m impi.v2.train ; -m impi.v2.ranker
  .venv\Scripts\python -m impi.live            # nightly live update (also --no-download, --export-only)
Root: python -m src.update_daily · python -m app.server · python -m pytest

-----------------------------------------------------------------------
8. HOW IT EVOLVED (decisions and turns in the conversation)
-----------------------------------------------------------------------
- Read Delivery Lens brief; verified it against files; found the failed 9 Oct run and the 5-day result change.
- Planned IMPI from the user's master spec; asked 4 questions (positioning, structure, timeline, UI timing).
- User said build everything in one go: gates enforced mechanically (pre-registration commit check,
  feature freeze hash, model freeze hash, one-time test lock).
- Download hit a stale connection (fixed) and a session restart (resumed); QC found 2 missed sessions;
  adjustment bugs fixed on real data; regime-feature gap fixed before registration.
- Results came out negative and were reported in full; paper draft written; user later wrote the paper
  in another chat.
- User asked to finish the terminal and maximise the model: v2 (post-hoc, honest), live track record, and
  terminal pages built and browser-tested.

-----------------------------------------------------------------------
9. STATUS AND WHAT'S LEFT
-----------------------------------------------------------------------
- Research, v2, live system, terminal: done and tested. Paper: written by the team in another chat
  (citations must be verified; Indian-market literature added by the team).
- Rough overall completion ~90-95%. Remaining: the forward track record matures (first outcomes ~10 trading
  days after 9 Oct 2026); final paper checks/formatting; optional future pre-registered study on new data.
- Maintenance notes: symbol-change and sector lists are not auto-refreshed (research reproducibility) —
  refresh manually if a stock is renamed; first nightly IMPI live run is Mon 12 Oct 2026 at 19:30 (10-25 min).

-----------------------------------------------------------------------
10. RULES FOR ANYONE CONTINUING
-----------------------------------------------------------------------
- Only real, dated numbers from the files. Label every result: pre-registered / exploratory / post-hoc / live.
- NEVER rerun P6 or tune anything on 2015-2026 data and call it a test. New hypotheses need a new
  pre-registration and new data (use the live track record).
- No result is "significant" unless it is (BH q <= 0.10 for the confirmatory family). H4 was a significant LOSS.
- Never present scores, probabilities or zones as buy/sell signals; always "research tool, not investment advice".
- Never call Ask Lens an AI/LLM; it is rule-based over the project's data.
- Keep the no-look-ahead rule, polite cached downloads, and the frozen research folders untouched.
- Don't build or change things without the team's approval; test in the browser before handing over.
