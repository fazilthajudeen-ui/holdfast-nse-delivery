HAND-OFF PROMPT — continue the "Real Time Orderflow Toxicity Detection" project (state on 14 Oct 2026, 14:15 IST)
==================================================================================================================

Paste this into the new chat together with PROJECT_PROMPT.md (same folder: C:\Users\fazil\delivery-movement-nse).
PROJECT_PROMPT.md is the full technical reference (data, methods, every result). This file is the story of what
the team asked for, what was decided,
and exactly where to pick up. Read both before doing anything.

HOW TO WORK WITH THIS TEAM (standing preferences)
- Propose first; build or change things only after an explicit go-ahead ("do it" / "finish it in one go" counts).
- Short, plain answers when they ask for "short" / "minimal"; they often paste outputs into slides and forms.
- Never invent numbers; label results pre-registered / exploratory / post-hoc / live.
- They have asked (12 Oct) to "make all hypotheses pass" / "change whatever it takes" — that was REFUSED as
  research misconduct, and must stay refused. Offer the honest route instead: new pre-registered hypotheses
  tested on new (live) data. Explain gently that "not supported" is a valid finding.
- When writing for slides/diary they prefer not to dwell on the negative side — describe aims, methods and
  what works, but never state anything false (e.g. never claim the v1 hypotheses passed).
- Test in the browser before handing over UI work. Never change Windows security settings.

WHAT WAS DONE, IN ORDER (10–14 Oct 2026)
10 Oct  Read the original brief; built and ran IMPI v1 (pre-registered, test set opened once): no hypothesis
        supported, XGBoost significantly worse than a simple rule; v2 models (post-hoc); live track record;
        web terminal (Terminal / Scanner / Market / Research).
10–11 Oct  Fact-checked the team's paper PDF ("Real Time Orderflow Toxicity Detection") against the result
        files and patched 6 numbers in place (n=165, 134–227/day, 841, 0.26, 75%, "2.9% overall"); kept title,
        format and references. Corrected copy: impi\paper\Real_Time_Orderflow_Toxicity_Detection_Paper.pdf.
11 Oct  "Improve the model": v3 — option-implied volatility features + 20-day targets, pre-registered first.
        Volatility model 0.743 -> 0.765 (post-hoc); 20-day breakout model has no skill. Live in the terminal.
        Fixed: Windows blocked an sklearn DLL (placeholder shim); terminal opened on last-viewed stock;
        auto-refreshing pages with "next update / overdue" label; nightly task hidden + 3 h limit.
11–12 Oct  Many explanation requests: what pre-registration means; why the 4 hypotheses failed; every figure,
        table and formula of the paper; viva answers; short conclusion + future work (time-cycle volatility
        forecasting); minimal answers to the 5 review questions.
12 Oct  Documents: completed Table IV image (impi\paper\Table_IV_backtests.png); corrected the PPT
        (impi\paper\Orderflow_Toxicity_Review.pptx: simple intro paragraph, removed Test-Set Performance slide,
        removed "Tools", removed conclusion point 3, fixed 5.84M / 134–227 / n=165); filled the Review 3 diary
        (impi\paper\EAD51806 Phase I project diary Review - 3.doc/.docx). Checked the downloaded paper PDF:
        numbers right, but wording garbled by a paraphrasing tool — told them not to submit it.
        Publication plan given (conference first, journal later; venues, missing references, power paragraph).
12 Oct  Team insight: "delivery doesn't mean price goes up — they could be selling". Also asked about Larry
        Williams' COT approach and loosening VCP. -> v4 pre-registered (commit d7cb5e3): D1 delivery -> move
        size, D2 signed delivery, H2′ all breakouts 20d, C1 Williams-style contrarian COT, V1 loose VCP.
        2023–26 check: D1 SUPPORTED (IC 0.046, q 0.005); V1 significant opposite (high delivery into a breakout
        = warning); others not supported. Built into the live system (logged nightly from 12 Oct) and the
        chart ("Signals" toggle: Acc / Dist / Move? / LVCP / Div; Market page C1 card + markers; Research table).
12 Oct  Ops fixes: laptop slept through 19:30 -> task also fires at log-on and on wake (scripts\should_run.py
        decides, data\logs\update.lock prevents overlap); Windows blocked pyarrow's Parquet DLL in the main
        Python -> src\_pyenv.py uses the IMPI venv's numpy/pandas/pyarrow; Delivery Lens now rebuilds if a
        previous rebuild crashed.
12 Oct  "Always shows the same stock with 99": data was updating (CGCL 72 -> 99.4 over two weeks); the terminal
        opened on the #1 stock and capped scores at 99. Now: one-decimal scores near the top; first open shows
        TODAY'S FRESHEST SIGNAL (High stock with the biggest score rise, labelled); last-viewed stock remembered.
        Web app also auto-starts the end-of-day update when due. Re-ran all hypotheses on 2024–26 (post-hoc,
        shown on the Research page): same picture — only D1 passes; H4 and V1 significant the wrong way.
14 Oct  Wrote PROJECT_PROMPT.md (full reference) and this hand-off.

CURRENT STATUS (14 Oct 2026, 14:15 IST)
- Delivery Lens data: as of 13 Oct 2026. IMPI live: as of 12 Oct; a catch-up run (started 11:33 by the
  wake/log-on trigger because the laptop missed 13 Oct 19:30) was in its final steps at 14:15. Its data step
  took 2.6 h (usual ~22 min; laptop probably slept/busy) — close to the task's 3-hour limit (14:33).
- Forward records: v2 since 9 Oct; v3 (F1–F5) since 9 Oct; v4 since 12 Oct. First verdicts ~Feb 2027.
- Tests: 56 IMPI + 41 app passing (12 Oct). Test server on http://localhost:8001 has everything; the
  port-8000 server (started in another chat) runs older code until restarted.

WHERE TO CONTINUE (in this order)
1. Verify the 14 Oct catch-up run: last line of data\logs\nightly.log should end "delivery_lens=0 impi_live=0";
   data\logs\last_success.txt dated 14 Oct; data\logs\update.lock gone; IMPI "as of" 13 or 14 Oct; v4
   days_logged 2+. If it was killed by the 3-hour limit, the 19:30 run will redo it — then consider (with the
   team's OK) raising the task limit to ~6 h, since sleep time counts against it.
2. Ask the team to restart the port-8000 server: python -m app.server (from the project folder).
3. Paper (biggest remaining deliverable): start from the clean PDF wording (not the downloaded paper PDF), fix 5.84M and
   "median turnover", use the full Table IV image, optionally add v4/D1 as a pre-registered follow-up and the
   power/CI paragraph, expand finance references (verify each), move to the venue template.
4. Diary: confirm each student's role. PPT: add a v4/D1 slide if they want.
5. Later: forward-test verdicts (~Feb 2027); optional new pre-registered study on time-cycle volatility
   forecasting; tidy overlapping "Move?" labels on the chart.

FILES TO KNOW
PROJECT_PROMPT.md (full reference) · impi\pre_registered_hypotheses.md / _v3.md / _v4.md · impi\decisions.md
(D-001..D-050) · impi\results\metrics\ (p6_test.json, v3_models.json, v4_check.json, subperiod_2024_2026.json)
· impi\data\track\ (predictions*.csv, summary*.json) · impi\paper\ (paper PDF, PPT, Table IV PNG, diary) ·
data\logs\ (nightly.log, last_success.txt) · scripts\ (run_daily.bat, schedule_daily.bat, should_run.py).
