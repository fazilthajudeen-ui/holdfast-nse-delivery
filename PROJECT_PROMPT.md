MASTER PROMPT — "REAL TIME ORDERFLOW TOXICITY DETECTION" (Delivery Lens + IMPI)
Complete project state, explanation and work log, as of 14 Oct 2026
=================================================================================

ROLE
You are taking over a finished, working final-year research project. Read this whole brief before answering.
Use it to explain the project, prepare presentations and viva answers, maintain or extend the code, or help
with the paper. Every number here comes from the project's own result files. Never invent numbers or
citations. If a figure is not here, compute it from the files or say it is unknown. Always say which date
a number is "as of", and label every result PRE-REGISTERED, EXPLORATORY, POST-HOC or LIVE.
Continue from PART 13 ("Open items") — that is where the work stopped.

=================================================================================
PART 1 — WHAT THE PROJECT IS
=================================================================================
- Title: "Real Time Orderflow Toxicity Detection". Software: Delivery Lens (original pilot) extended into
  IMPI — Indian Market Positioning Intelligence (research system + institutional-style web terminal).
- Student project. Course EAD51806 Phase I Project (Jun–Nov 2026).
- Domain: Artificial Intelligence and Machine Learning in Financial Markets (FinTech).
- Folder: C:\Users\fazil\delivery-movement-nse (Delivery Lens at root, not a git repo; IMPI in impi\ with
  its own git repo and Python venv impi\.venv).
- Research + education only. NOT investment advice. Scores are never buy/sell signals.

THE IDEA IN SIMPLE WORDS
Order flow is "toxic" when it comes from traders who know more than the people on the other side. On the NSE,
many traders believe two numbers published every evening reveal this informed money: the DELIVERY PERCENTAGE
(share of traded shares actually taken into demat) and changes in FUTURES POSITIONING (stock open interest
and FII/DII/Pro/Client positions). The belief: a breakout from a tight price range with high delivery and
supportive positioning is "real". IMPI rebuilds these footprints from official NSE files daily and tests the
belief fairly. "Real time" = each day's data is processed as soon as NSE publishes it (daily, not intraday).
Key insight added by the team (12 Oct 2026): DELIVERY IS DIRECTION-NEUTRAL — every delivered share has a
seller who delivered it, so high delivery shows conviction/participation, not who is in control
(accumulation OR distribution). This is what v4 (Part 8) tests, and it is the project's one positive result.

=================================================================================
PART 2 — HOW IT WORKS, STEP BY STEP (Fig. 1 of the paper)
=================================================================================
1. COLLECT official NSE end-of-day files every weekday evening: equity bhavcopy, security-wise delivery
   (sec_bhavdata_full; legacy bhavcopy + MTO before Oct 2019), F&O bhavcopy (futures, options, OI),
   participant-wise OI (market-wide totals only), F&O ban list, index closes incl. India VIX. Polite
   (>= 1 s apart), cached, resumable.
2. QUALITY CONTROL: reject stale holiday files (92), recover missed sessions (2), verify old vs new formats.
3. ADJUST prices AND quantities/OI for splits, bonuses, demergers (1,442 adjustments).
4. FEATURES: 80 features using only data published by 23:59 on day T (delivery z-score, volume, Amihud,
   Bollinger-width compression, VCP descriptors, stock OI change, COT-style positioning, basis, put-call
   ratio, FII/DII/Pro/Client positioning, regime). v3 adds 14 option-implied-volatility features.
5. BREAKOUTS: B1 = close above prior 20-day high; BROAD = B1 after compression (BB-width pct < 30 on >= 5 of
   10 days; main test group); STRICT = algorithmic VCP (2–4 shrinking contractions, 5–15 week base,
   breakout on >= 1.5x volume).
6. LABEL: triple barrier — success if entry + 2 ATR is hit before entry − 1 ATR within 10 days.
7. TEST AND MODEL: pre-registered hypothesis tests; XGBoost / random forest / logistic regression vs a
   fixed-weight composite; SHAP.
8. BACKTEST with Rs 10 lakh, max 20 positions, full Indian costs, vs Nifty 500.
9. LIVE: nightly scoring with frozen models/rules; predictions logged BEFORE outcomes; web terminal.

=================================================================================
PART 3 — KEY FORMULAS AND CONCEPTS (plain meaning)
=================================================================================
- Eq.(1) Triple barrier: y = 1 if E + 2A is hit before E − A within 10 days, else 0 (E = open of T+1,
  A = ATR14). Example: entry Rs 100, ATR Rs 3 -> success if Rs 106 comes before Rs 97. Same-day double touch
  = loss. Base success ~25%.
- Eq.(2) Delivery %: D = 100 x delivered qty / traded qty; models use its 20-day z-score vs own history.
- Eq.(3) logit Pr(y=1) = b0 + b1·D + b2·P + b3·(D x P) + controls; b3 = delivery-positioning interaction.
- p-value; q-value (Benjamini–Hochberg, corrects for several tests). PASS RULE (fixed in advance):
  predicted direction AND q <= 0.10. Groups with n < 30 get no p-value.
- AUC (0.5 = coin flip); DeLong test; top-decile precision; IC = rank correlation between a signal and the
  later outcome across stocks on each day.
- CAGR, Sharpe (above 6% risk-free), Max drawdown, Profit factor; Deflated Sharpe (corrects for 89 trials);
  Hansen SPA / White Reality Check.
- Pre-registration: hypotheses, tests and pass rule committed to git BEFORE results exist.
- Others: OI, COT-style index (position within 52-week range, 0–100), Amihud illiquidity, month-block
  bootstrap, purged CV with embargo, isotonic calibration, Brier score, SHAP, placebo test, Granger causality.

=================================================================================
PART 4 — DATA (official NSE archives, research set 1 Jan 2014 – 9 Oct 2026, frozen)
=================================================================================
3,161 trading days; 5,836,673 equity rows (5.84 million); 3,748 symbols incl. delisted; 354 F&O underlyings,
134–227 per day. Pre-2019 delivery rebuild: 0 mismatches on 12 overlap dates; F&O legacy vs UDiFF exact on
5 days; 92 stale files rejected; 2 sessions recovered (22 Mar 2014, 28 Feb 2015). 1,442 corporate actions:
841 NSE-confirmed, 468 gap-detected, 133 demergers. Participant positions are market-wide only. 11 Sep 2026
delivery file inconsistent -> missing. Live data continues daily after 9 Oct (12 Oct, 13 Oct ...).

=================================================================================
PART 5 — v1: THE PRE-REGISTERED STUDY (as of 10 Oct 2026)
=================================================================================
Design: train 1 Jun 2015–2020, validation 2021–22, test 1 Jan 2023–9 Oct 2026 (purged). Tier 1 = F&O stocks;
Tier 2 = EQ stocks with 60-day MEDIAN turnover > Rs 5 cr. Exclusions: ban T/T+1, results ±3 days, price < 20,
< 30 listing days, < 252 history days. Decision 23:59 T, entry T+1 open; truncation check 57/57, max diff 0.
Hypotheses: H1 delivery expansion -> more success; H2 price up + futures OI down -> less; H3 FII selling + DII
buying -> better gain/loss asymmetry; H4 XGBoost beats the composite (0.30 compression, 0.25 delivery,
0.25 positioning, 0.20 volume) in AUC and top-decile precision. 8 tests; STRICT groups n < 30 -> BH over 5.
Integrity: committed 15103f1 (10 Oct 2026 08:54 IST); test set run ONCE (lock ea49def, 09:23–09:29).
EVENTS (train/val/test/total, success): BROAD 2,102/978/1,796/4,876 (0.262); B1 5,075/1,979/4,273/11,327
(0.249); STRICT 38/15/29/82 (0.241).
EXPLORATORY (train+val): H1 27.3 vs 26.2% (p 0.51, q 0.94); H2 26.7 vs 26.8%; H3 1.03 vs 1.07; composite AUC
0.487/0.506; val AUC XGB 0.492, RF 0.487, logit 0.528; CV PR-AUC 0.31 vs base 0.26; XGB 14 trees depth 2;
SHAP delivery interactions = 0.
CONFIRMATORY TEST (run once) — NO HYPOTHESIS SUPPORTED: H1 24.8 vs 24.9% (−0.04 pp, p 0.99); H2 23.5 vs 26.3%
(−2.8 pp, p 0.17, q 0.29; −5.7 pp without pilot window, q 0.128); H3 1.01 vs 0.88 (p 0.38); H4 AUC XGBoost 0.460
vs composite 0.526 (DeLong p 0.005, q 0.024) — significant OPPOSITE direction; top decile 22.4% (n 165) vs 31.0%
(n 226), p 0.11. RF 0.466, logit 0.468, walk-forward 0.502; Brier 0.250 -> 0.195. Tier 2 H1 +0.8 pp (p 0.59).
LADDER: base 26.7% (159,972 stock-days); breakouts 24.9% (−1.9 pp, p 0.005); BROAD 24.8%; ML top decile
22.4%; composite >= 1: 41.4% but n = 29 (exploratory).
BACKTESTS (Table IV): All breakouts CAGR −26.8%, Sharpe −4.02, MaxDD −68.6%, PF 0.59, 2,736 trades; BROAD
−12.6%, −3.25, −39.2%, PF 0.60, 1,599; Composite 6.0%, −0.01, −1.0%, PF 1.04, 29 (n<30); ML top decile 4.3%,
−0.85, −2.3%, PF 0.64, 157; Nifty 500 9.7%, 0.32, −18.8%, n/a (buy & hold). SPA p 0.77, RC p 0.89, deflated
Sharpe 0 (89 trials). Image with all values: impi\paper\Table_IV_backtests.png.
ROBUSTNESS: H1 +1.0 pp 2015–19, −1.1 pp 2021–22; test-year success 28.8% / 24.0% / 24.3% / 17.4% (2023–26);
COVID removed H1 +0.9 pp; placebo: actual +1.05 pp beats all 20 shuffles (~1 pp real, too small to trade);
VCP grid 81 settings, of 72 with >= 30 events 75% positive (mean +2.8 pp); 104-week COT AUC 0.499 vs 0.506.
LEAD-LAG (exploratory): returns Granger-precede FII stock-futures positioning (p < 1e-7; FIIs follow price);
FII index-futures positioning precedes index returns (p ~0.001–0.02, also bidirectional); bottom-decile FII
stock-futures COT -> next 60 days Nifty 500 +6.4% vs 2.9% overall.
WHY NOT SUPPORTED (interpretation): delivery is direction-neutral; F&O stocks efficient; positioning only
market-wide and often hedging; breakouts underperform ordinary days; market changes (T+1 2023, F&O reforms
2024–25); Indian costs; with weak signal XGBoost overfits while a fixed rule cannot.

=================================================================================
PART 6 — v2 and v3 MODELS (POST-HOC; 2023+ was already used)
=================================================================================
Trained on all Tier-1 stock-days 2015–2022 (~285k rows), walk-forward CV 2018–22; 2023+ scored once.
v2 (10 Oct): 17 extra features, cross-sectional ranks. v3 (11 Oct; pre-registered first, commit 1b3df7b):
14 option-implied features (Black-76 IV from traded options, same-expiry future as forward; 30-day ATM IV,
skew, term structure, IV vs realised vol, option-volume ratios; legacy and UDiFF identical) + 20-day targets.
Adoption rule: v3 replaces v2 only if CV AUC >= +0.005 in >= 4/5 folds (met). Frozen sha256 662c977e...
| Target                 | v2 CV -> 2023+ | v3 CV -> 2023+ | Meaning |
| Volatility expansion   | 0.742 -> 0.743 | 0.753 -> 0.765 | useful; lowest decile 1.5% vs highest 54% |
| Beats median 10d       | 0.525 -> 0.524 | 0.533 -> 0.529 | small, not tradable after costs |
| Breakout 10d           | 0.538 -> 0.505 | 0.550 -> 0.524 | weak, unstable |
| Beats median 20d       | —              | 0.534 -> 0.528 | small |
| Breakout 20d           | —              | 0.542 -> 0.499 | no skill ("no edge" tag) |
Forward tests F1–F5 (v3 vol beats v2; AUCs > 0.5) — logged nightly since 9 Oct 2026, verdict ~Feb 2027.

=================================================================================
PART 7 — v4: RE-FRAMED HYPOTHESES (pre-registered 12 Oct 2026, commit d7cb5e3)
=================================================================================
Rules only, nothing trained (impi\pre_registered_v4.md, src\impi\v4\):
- D1 delivery -> MOVE SIZE: deliv_z20 vs move20 = |20-day return| ÷ (rv20·√(20/252)); daily Spearman IC.
- D2 SIGNED delivery -> direction: sds = mean over 10 days of (delivery % − prior-60-day median) x sign(daily
  return) (+ accumulation, − distribution) vs 20-day return.
- H2′ price up + OI down on ALL breakouts (B1) -> weaker 20-day triple-barrier success.
- C1 Larry Williams–style COT: FII stock-futures COT <= 10 -> higher next-60-day Nifty 500 (contrarian,
  market level). (Williams used per-market US COT; NSE gives only market-wide participant totals.)
- V1 looser VCP (k=5, depth 0.5, volume x1.25, base 20–60, chosen from the train+val grid): delivery
  expansion -> more 10-day success.
Two-sided p, block bootstraps (20 days for IC, 60 for C1), BH q <= 0.10 + direction.
RESULTS — 2023–26 check (run once; rules fixed in advance; previously used period, not clean):
| D1 | IC +0.046 (CI 0.028–0.063), positive on 69% of 915 days, p 0.001, q 0.005 | SUPPORTED |
| D2 | IC −0.008, p 0.29 | not supported |
| H2′| 29.1% vs 31.3% (−2.2 pp, right way), p 0.11, q 0.19 | not supported |
| C1 | 60-day Nifty 500 +2.5% after signal days vs +3.2% all days, p 0.67 | not supported |
| V1 | 19.7% vs 37.8% (−18.1 pp), q 0.011 | significant OPPOSITE (high delivery into a breakout = warning) |
POST-HOC 2024–26 RE-TEST (requested 12 Oct; exploratory, impi.research.subperiod, results\metrics\
subperiod_2024_2026.json): v1 — H1 q 0.98, H2 q 0.64 (wrong direction), H3 q 0.64, H4 AUC 0.457 vs 0.530
significant WRONG way (q 0.090), top decile q 0.64, STRICT insufficient. v4 — D1 PASSED again (IC 0.046,
q 0.005); D2, H2′, C1 not passed; V1 significant wrong way (q 0.071). Official verdicts unchanged.
LIVE FORWARD TEST (confirmatory): signals logged nightly before outcomes since 12 Oct 2026 (213 F&O stocks/day)
in impi\data\track\predictions_v4.csv (+ _market.csv); evaluated in summary_v4.json when: D1/D2 60 matured
dates (~Feb 2027); H2′ 1,000 breakouts (~late 2027); V1 30 per group (~2028); C1 750 days (~2029).

=================================================================================
PART 8 — LIVE SYSTEM, WEB APP, AUTOMATION
=================================================================================
- Run the app from the project folder: `python -m app.server` -> http://localhost:8000 (8001 = test copy).
- /terminal: watchlist (scores with one decimal near the top), chart with delivery bars, buyer zones, IMPI
  markers (● compression breakout, ■ VCP) and the "Signals" toggle (v4): Acc ▲ green / Dist ▼ red (signed
  delivery top/bottom 10% of the day), Move? ● orange (deliv_z20 >= 2), LVCP ■ purple (loose VCP), Div ●
  grey (breakout with falling OI). IMPI panel: 5 model probabilities (v3) with "no edge" tags, implied-vol
  snapshot, drivers, and "Delivery signals (v4)": expected move size rank, delivery flow, big-move watch.
  Opening rule: link -> last viewed stock -> TODAY'S FRESHEST SIGNAL (High stock with the biggest score rise
  since yesterday, labelled), so it no longer opens on the same 99-score leader every day.
- /scanner, /market (positioning, VIX, C1 card + C1 zone markers on the Nifty 500 chart), /research (v1 table
  with 2024–26 column, v2 vs v3 models, v4 table with 2023–26 check, 2024–26 re-test and live status,
  track records, backtests, figures). Ask Lens is RULE-BASED (not AI), now also states v4 move size/flow.
- Every page shows "Data as of <date> · next update <day>, 7:30 PM" / "updating now" / red "update overdue",
  polls every 5 minutes and reloads itself on a new trading day.
- AUTO UPDATE (end of day): Task Scheduler "NSE Delivery Update" runs scripts\run_daily.bat hidden, Mon–Fri
  19:30, PLUS at log-on and on wake from sleep (+2 min); 3-hour limit; runs on battery. scripts\should_run.py
  decides (due if last success is older than the latest weekday 19:30) and holds data\logs\update.lock so two
  runs never overlap. The web app also checks every 10 minutes and starts the update itself if due
  (--no-auto-update to disable). Logs: data\logs\nightly.log (healthy: "delivery_lens=0 impi_live=0"),
  last_success.txt, scheduler_stdout.txt, impi\data\logs\live_stdout.txt. Delivery Lens rebuilds if its
  scores are older than the newest raw file (protects against a crashed rebuild).
- Delivery Lens pilot: Conviction Score 0–100 (5-day delivery gap vs 60-day median, ranked over 250 days);
  as of 8 Oct 2026 High-Low +2.4 pts at 20 days (p 0.04), +1.7 pts at 60 days (p 0.49).

=================================================================================
PART 9 — ENVIRONMENT NOTES (Windows 11, 7.3 GB RAM, Python 3.12)
=================================================================================
Windows Smart App Control blocks some binaries; NEVER change security settings, route around them:
- pyarrow.dataset and statsmodels.api blocked -> code imports narrowly.
- 11 Oct: sklearn _datasets_pair.pyd blocked -> impi\src\impi\_sklearn_compat.py placeholder (v2 reproduces).
- 12 Oct: pyarrow 25.0.1 Parquet DLL in the MAIN Python blocked -> src\_pyenv.py puts the IMPI venv's
  numpy/pandas/pyarrow 26 first on sys.path for Delivery Lens and the web app (imported by src/ and app/
  __init__.py). If something stops loading, check Smart App Control first.
Tests (12 Oct): 56 IMPI + 41 app, all passing. Git (impi): 15103f1 v1 pre-reg · ea49def test lock · 1fc97ae v2
+ live · 1b3df7b v3 pre-reg · 522f7b3 v3 · d7cb5e3 v4 pre-reg · 9378244 v4 check · 83b3ae2 v4 live ·
0e67d29 2024–26 re-test. Decisions D-001..D-050 in impi\decisions.md.

=================================================================================
PART 10 — MODULES AND COMMANDS
=================================================================================
Modules: 1 Data ingestion · 2 Adjustment · 3 Feature engine (+ options) · 4 Event detection & labelling ·
5 Research & models (tests, ML, SHAP, backtests, v4 rules) · 6 Live system & web terminal.
Code: impi\src\impi\ (ingest, transforms, features, labels, research, models, backtest, validation, pipeline,
v2, v3, v4, live.py); app\ (server.py, impi_api.py, assistant.py, static\ incl. freshness.js); src\
(Delivery Lens); scripts\ (run_daily.bat, schedule_daily.bat, should_run.py).
Commands: impi\.venv\Scripts\python -m pytest · -m impi.live (--no-download / --export-only) ·
-m impi.v4.signals · root: python -m app.server · python -m src.update_daily (--force) · python -m pytest.

=================================================================================
PART 11 — DOCUMENTS (status 14 Oct 2026)
=================================================================================
- Paper: clean fact-checked copy impi\paper\Real_Time_Orderflow_Toxicity_Detection_Paper.pdf (still says
  5.83M -> 5.84M). the downloaded paper PDF has right numbers but its wording was garbled by a paraphrasing
  tool ("Amerind", "alien investors", "livery", "gunstock"...) — do not submit; also "average" -> "median"
  turnover; Table IV -> use the PNG. Neither version mentions v4/D1 yet.
- PPT: impi\paper\Orderflow_Toxicity_Review.pptx (22 slides, corrected, user's edits applied; no v4 yet).
- Review 3 diary: impi\paper\EAD51806 Phase I project diary Review - 3.doc/.docx (filled; roles are a
  suggested split to confirm; signatures/photos/contacts/attendance blank).
- Short conclusion: "We built IMPI to study whether NSE's daily delivery percentage and futures positioning
  reveal informed trading around stock breakouts, using a clean 2014–2026 dataset, hypotheses fixed in
  advance and full Indian costs." Future work: volatility forecasting based on time cycles (F&O expiry,
  earnings seasons, month-end, Budget/RBI events) with implied volatility, validated on fresh data.
- Publication plan: conference paper (rigorous pre-registered null + dataset + ML-vs-rule + D1 as a
  pre-registered follow-up), journal later with forward results. Venues to check: IEEE CIFEr, Springer
  CCIS/LNNS, Springer Computational Economics, Review of Quantitative Finance and Accounting, IEEE Access.
  Needed: clean text, power/CI paragraph (test-set H1 MDE ~6.3 pp, H1 95% CI ~ −4.9 to +4.9 pp), finance
  references (Kyle 1985; Easley, López de Prado & O'Hara 2012; Amihud 2002; Bailey & López de Prado 2014;
  White 2000; Hansen 2005; Benjamini & Hochberg 1995; Harvey, Liu & Zhu 2016 — verify each), code release,
  venue template; consider a title matching daily data.

=================================================================================
PART 12 — VIVA QUICK ANSWERS
=================================================================================
- Methods: triple barrier (fair, volatility-scaled), XGBoost/RF/logit vs simple rule + SHAP, pre-registration
  + BH + DeLong, cost-aware backtests. Dataset: official NSE, the only daily source of delivery/positioning.
- Why "not supported": failed the pre-set rule (direction AND q <= 0.10); H4 significant the wrong way.
- What worked: delivery predicts MOVE SIZE, not direction (D1, q 0.005, holds in 2024–26 too); the
  implied-volatility model forecasts volatility (AUC 0.765, post-hoc). High delivery into a loose-VCP
  breakout was a warning sign (V1, opposite direction).
- Complete? Yes — pipeline, models, backtests, nightly live system with forward tests and the terminal.
- Paper: written in IEEE format; final revision before submission.

=================================================================================
PART 13 — OPEN ITEMS (continue from here)
=================================================================================
1. Check the 14 Oct catch-up run finished: data\logs\nightly.log should end "delivery_lens=0 impi_live=0",
   last_success.txt updated, IMPI "as of" 13 Oct, v4 days_logged 2. (13 Oct 19:30 was missed — laptop off/
   asleep — and the wake/log-on trigger caught it up at 11:33 on 14 Oct.)
2. Restart the port-8000 server (another chat's) so it runs the current code.
3. Paper: clean the garbled wording, 5.84M, "median", full Table IV, optionally add v4/D1 as a pre-registered
   follow-up; diary roles; PPT has no v4 yet.
4. Forward verdicts: v3 F1–F5 and v4 D1/D2 ~Feb 2027 (60 matured dates); later H2′, V1, C1.
5. Possible future work (needs approval + new pre-registration): time-cycle volatility forecasting; tidy
   overlapping "Move?" chart labels.

=================================================================================
PART 14 — RULES FOR ANYONE CONTINUING
=================================================================================
- Only real, dated numbers from the files; label every result.
- NEVER rerun the v1 test or v4 check, tune on 2015–2026 data and call it a test, or change results/rules to
  make hypotheses "pass" (the team asked; it was refused). New ideas need a new pre-registration + new data.
- Nothing is "significant" unless BH q <= 0.10 in its family. Never present scores as buy/sell signals;
  never call Ask Lens an AI. Keep no-look-ahead, polite downloads, frozen research folders. Don't change the
  project without the team's approval; test in the browser before handing over. Never change security settings.
