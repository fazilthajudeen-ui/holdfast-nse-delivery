"""Generate paper/numbers.tex: one \\newcommand per number used in the paper text, read ONLY from result
JSON (locked: results/metrics/*.json; post-hoc: results/posthoc/*.json). Also writes
paper/numbers_sources.csv (macro, value, source file, key) for the number audit.

    .venv/Scripts/python.exe scripts/build_macros.py
Placebo macros print a visible [pending] placeholder until results/posthoc/A_placebo.json holds 1,000 draws.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MET, PH, PAPER = ROOT / "results" / "metrics", ROOT / "results" / "posthoc", ROOT / "paper"
J = lambda p: json.loads(p.read_text(encoding="utf-8"))
MACROS: list[tuple[str, str, str, str]] = []


def name_ok(n):
    assert re.fullmatch(r"[A-Za-z]+", n), n
    assert n not in {m[0] for m in MACROS}, f"duplicate {n}"
    return n


def M(name, text, src, key):
    MACROS.append((name_ok(name), str(text), src, key))


def minus(s: str) -> str:
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s      # works in text and in math mode


def f(x, d=2, sign=False, pct=False):
    """Number -> LaTeX text. pct multiplies by 100."""
    v = 100 * x if pct else x
    s = f"{v:+,.{d}f}" if sign else f"{v:,.{d}f}"
    return minus(s)


def P(p):
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def main():
    p6, p3, p5, p2 = J(MET / "p6_test.json"), J(MET / "p3_hypotheses.json"), J(MET / "p5_ml.json"), J(MET / "p2_summary.json")
    qc, adj, g1, p7 = J(MET / "qc_report.json"), J(MET / "adjustment_qc.json"), J(MET / "p1_g1_check.json"), J(MET / "p7_robustness.json")
    s0, facts = J(PH / "s0_supplementary_audit.json"), J(PH / "paper_facts.json")
    A, B, C, D, E, F, G = (J(PH / x) for x in ["A_placebo.json", "B_breakeven.json", "C_spec_curve.json", "D_bayes.json",
                                               "E_clustered_logit.json", "F_h1_by_year.json", "G_breakout_ties.json"])
    m6, m3, mqc = "results/metrics/p6_test.json", "results/metrics/p3_hypotheses.json", "results/metrics/qc_report.json"

    # ---------------- dataset
    M("NTradingDays", f"{qc['calendar']['trading_days']:,}", mqc, "calendar.trading_days")
    M("DataStart", qc["calendar"]["first"], mqc, "calendar.first")
    M("DataEnd", qc["calendar"]["last"], mqc, "calendar.last")
    M("NEquityRowsM", f"{qc['rows']['equity_rows'] / 1e6:.2f}", mqc, "rows.equity_rows (/1e6)")
    M("NEquityRows", f"{qc['rows']['equity_rows']:,}", mqc, "rows.equity_rows")
    M("NSymbols", f"{facts['n_symbols_equity_store']:,}", "results/posthoc/paper_facts.json", "n_symbols_equity_store")
    M("NFOUnderlyings", facts["n_fo_underlyings_ever"], "results/posthoc/paper_facts.json", "n_fo_underlyings_ever")
    M("FOPerDayMin", facts["fo_underlyings_per_day_min"], "results/posthoc/paper_facts.json", "fo_underlyings_per_day_min")
    M("FOPerDayMax", facts["fo_underlyings_per_day_max"], "results/posthoc/paper_facts.json", "fo_underlyings_per_day_max")
    M("NStaleFiles", qc["stale_files_rejected"], mqc, "stale_files_rejected")
    ov = qc["legacy_vs_sec_overlap"]
    M("NDelivOverlapDates", len(ov), mqc, "len(legacy_vs_sec_overlap)")
    M("NDelivOverlapMismatch", sum(o["deliv_qty_mismatch"] + o["close_mismatch"] + o["volume_mismatch"] for o in ov), mqc,
      "sum(legacy_vs_sec_overlap.*_mismatch)")
    M("NFOOverlapDays", len(qc["fo_legacy_vs_udiff"]), mqc, "len(fo_legacy_vs_udiff)")
    ma = "results/metrics/adjustment_qc.json"
    mt = adj["methods"]
    M("NAdjust", f"{adj['actions_applied']:,}", ma, "actions_applied")
    M("NAdjustNSE", mt["nse_confirmed_gap"] + mt["nse_confirmed_gap_close"], ma, "methods.nse_confirmed_gap(+_close)")
    M("NAdjustGap", mt["gap_snap"], ma, "methods.gap_snap")
    M("NAdjustDemerger", mt["demerger_gap"] + mt["demerger_gap_close"], ma, "methods.demerger_gap(+_close)")
    M("NGOneChecks", g1["n_checks"], "results/metrics/p1_g1_check.json", "n_checks")
    M("GOneMaxDiff", f"{max(t['max_abs_diff'] for t in g1['truncation']):g}", "results/metrics/p1_g1_check.json", "max(truncation.max_abs_diff)")
    M("NFeatures", p2["features_v1"], "results/metrics/p2_summary.json", "features_v1")

    # ---------------- events
    c = p2["tier1_eligible_counts"]
    for fam, tag in [("broad", "Broad"), ("all", "All"), ("strict", "Strict")]:
        for sp, st in [("train", "Train"), ("validation", "Val"), ("test", "Test")]:
            M(f"N{tag}{st}", f"{c[fam][sp]:,}", "results/metrics/p2_summary.json", f"tier1_eligible_counts.{fam}.{sp}")
        M(f"N{tag}Total", f"{c[fam]['train'] + c[fam]['validation'] + c[fam]['test']:,}", "results/metrics/p2_summary.json",
          f"sum tier1_eligible_counts.{fam}.train+validation+test")
        M(f"Succ{tag}", f"{p2['class_balance'][fam]['label_10']:.3f}", "results/metrics/p2_summary.json", f"class_balance.{fam}.label_10")

    # ---------------- train/validation (exploratory, P3/P5)
    tv = {(t["hypothesis"], t["family"]): t for t in p3["tests"]}
    h1, h2, h3 = tv[("H1", "BROAD")], tv[("H2", "BROAD")], tv[("H3", "BROAD")]
    M("TvHOneExp", f(h1["p1"], 1, pct=True), m3, "tests[H1,BROAD].p1")
    M("TvHOneRest", f(h1["p2"], 1, pct=True), m3, "tests[H1,BROAD].p2")
    M("TvHOneDiff", f(h1["diff"], 1, True, True), m3, "tests[H1,BROAD].diff")
    M("TvHOneP", P(h1["p"]), m3, "tests[H1,BROAD].p")
    M("TvHOneQ", f"{h1['bh_q']:.2f}", m3, "tests[H1,BROAD].bh_q")
    M("TvHTwoExp", f(h2["p1"], 1, pct=True), m3, "tests[H2,BROAD].p1")
    M("TvHTwoRest", f(h2["p2"], 1, pct=True), m3, "tests[H2,BROAD].p2")
    M("TvHTwoP", P(h2["p"]), m3, "tests[H2,BROAD].p")
    M("TvHThreeExp", f(h3["asym_exposed"], 2), m3, "tests[H3,BROAD].asym_exposed")
    M("TvHThreeRest", f(h3["asym_rest"], 2), m3, "tests[H3,BROAD].asym_rest")
    M("TvHThreeP", P(h3["p"]), m3, "tests[H3,BROAD].p")
    m5 = "results/metrics/p5_ml.json"
    M("ValAUCxgb", f"{p5['xgb_val']['auc']:.3f}", m5, "xgb_val.auc")
    M("ValAUCrf", f"{p5['rf_val']['auc']:.3f}", m5, "rf_val.auc")
    M("ValAUClogit", f"{p5['logit_val']['auc']:.3f}", m5, "logit_val.auc")
    M("ValAUCcomp", f"{p5['composite_val_auc']:.3f}", m5, "composite_val_auc")
    M("XgbTrees", p5["xgb_val"]["best_iteration"] + 1, m5, "xgb_val.best_iteration + 1")
    M("XgbDepth", p5["best_params"]["xgb"]["max_depth"], m5, "best_params.xgb.max_depth")

    # ---------------- confirmatory (test, locked P6)
    cf = {(t["hypothesis"], t["family"]): t for t in p6["confirmatory"]}
    cn = {(t["hypothesis"], t["family"]): t for t in p6["confirmatory_without_pilot_window"]}
    b1, b2, b3, a4, t4 = cf[("H1", "BROAD")], cf[("H2", "BROAD")], cf[("H3", "BROAD")], cf[("H4_auc", "BROAD")], cf[("H4_top_decile", "BROAD")]
    M("HOneExp", f(b1["p1"], 1, pct=True), m6, "confirmatory[H1,BROAD].p1")
    M("HOneRest", f(b1["p2"], 1, pct=True), m6, "confirmatory[H1,BROAD].p2")
    M("HOneDiff", f(b1["diff"], 2, True, True), m6, "confirmatory[H1,BROAD].diff")
    M("HOneP", P(b1["p"]), m6, "confirmatory[H1,BROAD].p")
    M("HOneQ", f"{b1['bh_q']:.3f}", m6, "confirmatory[H1,BROAD].bh_q")
    M("HTwoExp", f(b2["p1"], 1, pct=True), m6, "confirmatory[H2,BROAD].p1")
    M("HTwoRest", f(b2["p2"], 1, pct=True), m6, "confirmatory[H2,BROAD].p2")
    M("HTwoDiff", f(b2["diff"], 1, True, True), m6, "confirmatory[H2,BROAD].diff")
    M("HTwoP", P(b2["p"]), m6, "confirmatory[H2,BROAD].p")
    M("HTwoQ", f"{b2['bh_q']:.3f}", m6, "confirmatory[H2,BROAD].bh_q")
    M("HThreeExp", f(b3["asym_exposed"], 2), m6, "confirmatory[H3,BROAD].asym_exposed")
    M("HThreeRest", f(b3["asym_rest"], 2), m6, "confirmatory[H3,BROAD].asym_rest")
    M("HThreeP", P(b3["p"]), m6, "confirmatory[H3,BROAD].p")
    M("HThreeQ", f"{b3['bh_q']:.3f}", m6, "confirmatory[H3,BROAD].bh_q")
    M("AUCxgb", f"{p6['ml_test']['xgb']['auc']:.3f}", m6, "ml_test.xgb.auc")
    M("AUCrf", f"{p6['ml_test']['rf']['auc']:.3f}", m6, "ml_test.rf.auc")
    M("AUClogit", f"{p6['ml_test']['logit']['auc']:.3f}", m6, "ml_test.logit.auc")
    M("AUCcomp", f"{p6['ml_test']['composite_auc']:.3f}", m6, "ml_test.composite_auc")
    M("HFourDiff", f(a4["diff"], 3, True), m6, "confirmatory[H4_auc].diff")
    M("HFourSE", f"{a4['se']:.3f}", m6, "confirmatory[H4_auc].se")
    M("HFourP", P(a4["p"]), m6, "confirmatory[H4_auc].p")
    M("HFourQ", f"{a4['bh_q']:.3f}", m6, "confirmatory[H4_auc].bh_q")
    M("TopMLPrec", f(t4["ml_precision"], 1, pct=True), m6, "confirmatory[H4_top_decile].ml_precision")
    M("TopCompPrec", f(t4["composite_precision"], 1, pct=True), m6, "confirmatory[H4_top_decile].composite_precision")
    M("TopMLN", t4["n_ml"], m6, "confirmatory[H4_top_decile].n_ml")
    M("TopP", P(t4["p"]), m6, "confirmatory[H4_top_decile].p")
    sp = t4["return_spread_ml_minus_composite"]
    M("TopSpread", f(sp["estimate"], 2, True, True), m6, "confirmatory[H4_top_decile].return_spread.estimate")
    M("TopSpreadLo", f(sp["ci_low"], 2, True, True), m6, "...return_spread.ci_low")
    M("TopSpreadHi", f(sp["ci_high"], 2, True, True), m6, "...return_spread.ci_high")
    M("BrierRaw", f"{p6['ml_test']['xgb']['brier_raw']:.3f}", m6, "ml_test.xgb.brier_raw")
    M("BrierCal", f"{p6['ml_test']['xgb']['brier_calibrated']:.3f}", m6, "ml_test.xgb.brier_calibrated")
    M("WFAUC", f"{p6['walk_forward']['oos_auc']:.3f}", m6, "walk_forward.oos_auc")
    M("WFMonths", p6["walk_forward"]["months"], m6, "walk_forward.months")
    n2, n4 = cn[("H2", "BROAD")], cn[("H4_auc", "BROAD")]
    M("NPHTwoDiff", f(n2["diff"], 1, True, True), m6, "confirmatory_without_pilot_window[H2,BROAD].diff")
    M("NPHTwoP", P(n2["p"]), m6, "confirmatory_without_pilot_window[H2,BROAD].p")
    M("NPHTwoQ", f"{n2['bh_q']:.3f}", m6, "confirmatory_without_pilot_window[H2,BROAD].bh_q")
    M("NPHFourDiff", f(n4["diff"], 3, True), m6, "confirmatory_without_pilot_window[H4_auc].diff")
    M("NPHFourP", P(n4["p"]), m6, "confirmatory_without_pilot_window[H4_auc].p")
    M("BHm", s0["1_bh_family"]["rows"][0]["m"], "results/posthoc/s0_supplementary_audit.json", "1_bh_family.rows[0].m")
    M("BaseRateTest", f(p6["ml_test"]["base_rate"], 3), m6, "ml_test.base_rate")
    t2 = p6["tier2_replication_H1"]["broad"]
    M("TTwoDiff", f(t2["diff"], 1, True, True), m6, "tier2_replication_H1.broad.diff")
    M("TTwoP", P(t2["p"]), m6, "tier2_replication_H1.broad.p")
    pr = p6["power_rule_H1_strict_tier1_or_2"]
    M("PowRuleN", pr["n"], m6, "power_rule_H1_strict_tier1_or_2.n")
    M("PowRuleDiff", f(pr["diff"], 1, True, True), m6, "power_rule_H1_strict_tier1_or_2.diff")
    M("PowRuleP", P(pr["p"]), m6, "power_rule_H1_strict_tier1_or_2.p")

    # ---------------- ladder + backtests
    lb = {r["rung"]: r for r in p6["ladder_broad"]}
    M("LadBaseN", f"{lb['B0_base']['n']:,}", m6, "ladder_broad[B0_base].n")
    M("LadBase", f(lb["B0_base"]["success_rate"], 1, pct=True), m6, "ladder_broad[B0_base].success_rate")
    M("LadAll", f(lb["B1_all_breakouts"]["success_rate"], 1, pct=True), m6, "ladder_broad[B1].success_rate")
    M("LadAllDiff", f(lb["B1_all_breakouts"]["vs_previous"]["success_diff"], 1, True, True), m6, "ladder_broad[B1].vs_previous.success_diff")
    M("LadAllP", P(lb["B1_all_breakouts"]["vs_previous"]["success_p"]), m6, "ladder_broad[B1].vs_previous.success_p")
    M("LadBroad", f(lb["B2b_broad"]["success_rate"], 1, pct=True), m6, "ladder_broad[B2b].success_rate")
    M("LadComp", f(lb["B3b_broad_composite"]["success_rate"], 1, pct=True), m6, "ladder_broad[B3b].success_rate")
    M("LadCompN", lb["B3b_broad_composite"]["n"], m6, "ladder_broad[B3b].n")
    bt = {b["variant"]: b for b in p6["backtests"]}
    for v, tag in [("B1_all_breakouts", "All"), ("B2b_broad", "Broad"), ("B3b_broad_composite", "Comp"), ("B4b_broad_ml_top_decile", "ML")]:
        M(f"Cagr{tag}", f(bt[v]["cagr"], 1, True, True), m6, f"backtests[{v}].cagr")
        M(f"Trades{tag}", f"{bt[v]['n_trades']:,}", m6, f"backtests[{v}].n_trades")
        M(f"MaxDD{tag}", f(bt[v]["max_dd"], 1, pct=True), m6, f"backtests[{v}].max_dd")
    bm = p6["benchmark_nifty500_test"]
    M("CagrNifty", f(bm["cagr"], 1, True, True), m6, "benchmark_nifty500_test.cagr")
    M("SPAp", f"{p6['spa']['spa_p_consistent']:.2f}", m6, "spa.spa_p_consistent")
    M("RCp", f"{p6['spa']['white_rc_p']:.2f}", m6, "spa.white_rc_p")
    M("DSRtrials", list(p6["deflated_sharpe"].values())[0]["n_trials_results_log"], m6, "deflated_sharpe.*.n_trials_results_log")

    # ---------------- integrity / timeline (supplementary audit, from git + run logs)
    tl = s0["10_timeline"]
    ms0 = "results/posthoc/s0_supplementary_audit.json"
    lock = tl["test_lock"]["attempts"][0]
    M("TrialsBefore", tl["results_log_rows_before_test_run_started"], ms0, "10_timeline.results_log_rows_before_test_run_started")
    M("TrialsTestRun", tl["results_log_rows_through_P6"] - tl["results_log_rows_before_test_run_started"], ms0, "through_P6 - before")
    M("TrialsThroughPSix", tl["results_log_rows_through_P6"], ms0, "10_timeline.results_log_rows_through_P6")
    M("PreregCommit", tl["files_first_committed"]["pre_registered_hypotheses.md"].split()[0], ms0, "files_first_committed[pre_registered_hypotheses.md]")
    M("PreregTime", tl["files_first_committed"]["pre_registered_hypotheses.md"].split()[1][11:16], ms0, "files_first_committed[pre_registered_hypotheses.md]")
    M("LockHead", lock["git_commit"], ms0, "10_timeline.test_lock.attempts[0].git_commit")
    M("LockStart", lock["started"][11:16], ms0, "10_timeline.test_lock.attempts[0].started")
    M("LockCommit", tl["files_first_committed"]["results/test_lock.json"].split()[0], ms0, "files_first_committed[results/test_lock.json]")
    M("LockCommitTime", tl["files_first_committed"]["results/test_lock.json"].split()[1][11:16], ms0, "files_first_committed[results/test_lock.json]")

    # ---------------- post-hoc: supplementary audit
    tost = {(r["set"], r["family"], r["hypothesis"]): r for r in s0["2_tost"]["rows"]}
    for (st, h), tag in [(("full", "H1"), "HOne"), (("full", "H2"), "HTwo"), (("without_pilot", "H1"), "NPHOne"), (("without_pilot", "H2"), "NPHTwo")]:
        r = tost[(st, "BROAD", h)]
        M(f"Tost{tag}Lo", f(r["ci90_wald"][0], 1, True, True), ms0, f"2_tost[{st},BROAD,{h}].ci90_wald[0]")
        M(f"Tost{tag}Hi", f(r["ci90_wald"][1], 1, True, True), ms0, f"2_tost[{st},BROAD,{h}].ci90_wald[1]")
        M(f"Tost{tag}P", f"{r['p_tost']:.2f}", ms0, f"2_tost[{st},BROAD,{h}].p_tost")
    mde = {(r["family"], r["hypothesis"]): r for r in s0["3_mde"]["rows"]}
    M("MDEHOne", f"{mde[('BROAD', 'H1')]['mde_pp_prereg_method']:.2f}", ms0, "3_mde[BROAD,H1].mde_pp_prereg_method")
    M("MDEHTwo", f"{mde[('BROAD', 'H2')]['mde_pp_prereg_method']:.2f}", ms0, "3_mde[BROAD,H2].mde_pp_prereg_method")
    ex = [r for r in s0["4_h1_ex_expiry"]["rows"] if r["family"] == "BROAD"]
    e0 = [r for r in ex if r["window"] == "5_sessions_before_expiry_day"][0]
    M("ExpN", s0["4_h1_ex_expiry"]["n_monthly_expiries_in_test"], ms0, "4_h1_ex_expiry.n_monthly_expiries_in_test")
    M("ExpRemoved", e0["n_removed"], ms0, "4_h1_ex_expiry.rows[5_sessions,BROAD].n_removed")
    M("ExpDiff", f(e0["diff"], 1, True, True), ms0, "4_h1_ex_expiry.rows[5_sessions,BROAD].diff")
    M("ExpP", P(e0["p"]), ms0, "4_h1_ex_expiry.rows[5_sessions,BROAD].p")
    M("ExpPMin", P(min(r["p"] for r in ex)), ms0, "min p over 3 window variants")
    psi = {r["feature"]: r["psi"] for r in s0["5_auc_psi"]["psi_top10_shap"]}
    M("PSIpcr", f"{psi['log_pcr_oi']:.2f}", ms0, "5_auc_psi.psi_top10_shap[log_pcr_oi].psi")
    M("PSIclient", f"{psi['client_fut_idx_cot']:.2f}", ms0, "5_auc_psi.psi_top10_shap[client_fut_idx_cot].psi")
    M("PSInModerate", sum(1 for v in psi.values() if 0.10 <= v <= 0.25), ms0, "count 0.10<=psi<=0.25")
    M("PSInLarge", sum(1 for v in psi.values() if v > 0.25), ms0, "count psi>0.25")
    br = s0["6_brier"]
    M("BrierConstVal", f"{br['brier_constant_at_validation_base_rate']:.3f}", ms0, "6_brier.brier_constant_at_validation_base_rate")
    M("BrierConstTest", f"{br['brier_constant_at_test_base_rate']:.3f}", ms0, "6_brier.brier_constant_at_test_base_rate")
    M("IncompleteBroad", s0["8_incomplete_labels"]["all_tier1_eligible_with_missing_label_10"]["broad"], ms0, "8_incomplete_labels...broad")
    gap = s0["9c_165_vs_157"]
    M("GapEvents", gap["ml_top_decile_events"], ms0, "9c_165_vs_157.ml_top_decile_events")
    M("GapTrades", gap["trades"], ms0, "9c_165_vs_157.trades")
    M("GapSlotsFull", gap["reasons"].get("all 20 slots full", 0), ms0, "9c_165_vs_157.reasons[all 20 slots full]")

    # ---------------- post-hoc B
    mb = "results/posthoc/B_breakeven.json"
    be = B["break_even"]
    M("RoundTrip", f"{be['round_trip_cost_pct_weighted']:.2f}", mb, "break_even.round_trip_cost_pct_weighted")
    M("RTLarge", f"{be['cost_components_by_bucket']['large']['round_trip_pct']:.2f}", mb, "cost_components_by_bucket.large.round_trip_pct")
    M("RTSmall", f"{be['cost_components_by_bucket']['small']['round_trip_pct']:.2f}", mb, "cost_components_by_bucket.small.round_trip_pct")
    M("AvgWin", f(be["avg_success_return_pct"], 2, True), mb, "break_even.avg_success_return_pct")
    M("AvgLoss", f(be["avg_failure_return_pct"], 2, True), mb, "break_even.avg_failure_return_pct")
    M("WinLossSpread", f"{be['spread_pp']:.2f}", mb, "break_even.spread_pp")
    M("BreakEven", f"{be['break_even_success_rate_change_pp']:.1f}", mb, "break_even.break_even_success_rate_change_pp")
    rows = {(r["test"], r["family"], r["set"]): r for r in B["rows"] if "note" not in r}
    for (t, st), tag in [(("H1 delivery expansion", "full"), "HOne"), (("H2 price up, OI down", "full"), "HTwo"),
                         (("H1 delivery expansion", "without_pilot"), "NPHOne"), (("H2 price up, OI down", "without_pilot"), "NPHTwo"),
                         (("ML top decile (XGBoost)", "full"), "ML"), (("Composite top decile (H4 cut)", "full"), "CompTop"),
                         (("Composite rule (score >= 1.0)", "full"), "CompRule")]:
        r = rows[(t, "BROAD", st)]
        M(f"BE{tag}Diff", f(r["diff"], 1, True, True), mb, f"rows[{t},BROAD,{st}].diff")
        M(f"BE{tag}Lo", f(r["ci95"][0], 1, True, True), mb, f"rows[{t},BROAD,{st}].ci95[0]")
        M(f"BE{tag}Hi", f(r["ci95"][1], 1, True, True), mb, f"rows[{t},BROAD,{st}].ci95[1]")
        M(f"BE{tag}BLo", f(r["ci95_block"][0], 1, True, True), mb, f"rows[{t},BROAD,{st}].ci95_block[0]")
        M(f"BE{tag}BHi", f(r["ci95_block"][1], 1, True, True), mb, f"rows[{t},BROAD,{st}].ci95_block[1]")
        M(f"BE{tag}N", r["n1"], mb, f"rows[{t},BROAD,{st}].n1")
    nb = sum(1 for r in B["rows"] if "note" not in r and r["family"] == "BROAD" and r["test"].startswith(("H1", "H2")) and r["below_break_even_ci95"])
    M("BEnBelow", nb, mb, "count BROAD H1/H2 rows with below_break_even_ci95")

    # ---------------- post-hoc C
    mc = "results/posthoc/C_spec_curve.json"
    for h, tag in [("H1", "One"), ("H2", "Two")]:
        s = C["summary"][h]
        M(f"Spec{tag}N", s["n_specs"], mc, f"summary.{h}.n_specs")
        M(f"Spec{tag}Pred", f(s["share_predicted_direction"], 0, pct=True), mc, f"summary.{h}.share_predicted_direction")
        M(f"Spec{tag}Sig", s["n_significant"], mc, f"summary.{h}.n_significant")
        M(f"Spec{tag}SigPct", f(s["share_significant_5pct"], 0, pct=True), mc, f"summary.{h}.share_significant_5pct")
        M(f"Spec{tag}Opp", s["n_sig_opposite_direction"], mc, f"summary.{h}.n_sig_opposite_direction")
        M(f"Spec{tag}BH", s["n_bh_q10_within_hypothesis"], mc, f"summary.{h}.n_bh_q10_within_hypothesis")
        M(f"Spec{tag}Median", f(s["median_effect_pp"], 2, True), mc, f"summary.{h}.median_effect_pp")
        M(f"Spec{tag}Max", f(s["max_effect_predicted_dir_pp"], 2), mc, f"summary.{h}.max_effect_predicted_dir_pp")
        M(f"Spec{tag}CIAbove", s["n_upper_ci95_above_break_even"], mc, f"summary.{h}.n_upper_ci95_above_break_even")
        M(f"Spec{tag}Frozen", f(s["frozen_spec_effect_pp"], 2, True), mc, f"summary.{h}.frozen_spec_effect_pp")
        M(f"Spec{tag}Rank", s["frozen_rank_in_predicted_direction"], mc, f"summary.{h}.frozen_rank_in_predicted_direction")

    # ---------------- post-hoc D
    md = "results/posthoc/D_bayes.json"
    dd = {(r["family"], r["set"], r["hypothesis"], r["prior"]): r for r in D["rows"]}
    for h, tag in [("H1", "One"), ("H2", "Two")]:
        for pr_, pt in [("default Beta(1,1)", "Uni"), ("skeptical Beta(100*pi,100*(1-pi))", "Skep"),
                        ("default logOR~N(0,1)", "LogDef"), ("skeptical logOR~N(0,0.25^2)", "LogSkep")]:
            for st, stag in [("full", ""), ("without_pilot", "NP")]:
                M(f"BF{stag}{tag}{pt}", f"{dd[('BROAD', st, h, pr_)]['BF01']:.2f}", md, f"rows[BROAD,{st},{h},{pr_}].BF01")

    # ---------------- post-hoc E
    me = "results/posthoc/E_clustered_logit.json"
    ee = {(r["hypothesis"], r["beta"]): r for r in E["rows"] if r["family"] == "BROAD" and r["set"] == "full"
          and r["se_type"] == "two_way_date_symbol"}
    for h, tag in [("H1", "One"), ("H2", "Two")]:
        for b, bt_ in [("beta1", "BOne"), ("beta2", "BTwo"), ("beta3", "BThree")]:
            r = ee[(h, b)]
            M(f"E{tag}{bt_}", f(r["coef"], 3, True), me, f"rows[BROAD,full,two_way,{h},{b}].coef")
            M(f"E{tag}{bt_}Lo", f(r["ci95"][0], 3, True), me, f"...{h},{b}.ci95[0]")
            M(f"E{tag}{bt_}Hi", f(r["ci95"][1], 3, True), me, f"...{h},{b}.ci95[1]")
            M(f"E{tag}{bt_}P", P(r["p"]), me, f"...{h},{b}.p")
    mm = [r for r in E["rows"] if r["family"] == "BROAD" and r["set"] == "full" and r["se_type"] == "month"]
    M("EMonthMatches", "exactly" if all(abs(r["p"] - r["locked_p6_p_month"]) < 1e-9 for r in mm) else "NOT", me,
      "month-clustered p == locked p6 p (all BROAD full terms)")

    # ---------------- post-hoc F, G
    mf = "results/posthoc/F_h1_by_year.json"
    yrs = [r for r in F["rows"] if r["family"] == "BROAD" and not r["period"].startswith("pooled")]
    M("FYears", len(yrs), mf, "rows[BROAD, year]")
    M("FYearsSig", sum(1 for r in yrs if r["p"] is not None and r["p"] < 0.05), mf, "count year rows with p<0.05")
    M("FFirstYear", yrs[0]["period"], mf, "rows[BROAD][0].period")
    M("FLastYear", yrs[-1]["period"], mf, "rows[BROAD][-1].period")
    mg = "results/posthoc/G_breakout_ties.json"
    M("GTieShareLoss", f(G["B1_breakout_events"]["tie_share_of_losses"], 2, pct=True), mg, "B1_breakout_events.tie_share_of_losses")
    M("GTieExplained", f(G["gap_decomposition_pp"]["share_of_gap_explained_by_ties_excluded"], 1, pct=True), mg,
      "gap_decomposition_pp.share_of_gap_explained_by_ties_excluded")
    M("GDayOneBO", f(G["B1_breakout_events"]["day1_stop_share"], 1, pct=True), mg, "B1_breakout_events.day1_stop_share")
    M("GDayOneNB", f(G["non_breakout_days"]["day1_stop_share"], 1, pct=True), mg, "non_breakout_days.day1_stop_share")
    M("GRangeBO", f"{G['B1_breakout_events']['mean_day1_range_atr']:.2f}", mg, "B1_breakout_events.mean_day1_range_atr")
    M("GRangeNB", f"{G['non_breakout_days']['mean_day1_range_atr']:.2f}", mg, "non_breakout_days.mean_day1_range_atr")
    M("GGapBO", f"{G['B1_breakout_events']['mean_gap_atr']:.2f}", mg, "B1_breakout_events.mean_gap_atr")
    M("GGapNB", f"{G['non_breakout_days']['mean_gap_atr']:.2f}", mg, "non_breakout_days.mean_gap_atr")
    M("GATRBO", f"{G['B1_breakout_events']['mean_atr_pct']:.2f}", mg, "B1_breakout_events.mean_atr_pct")
    M("GATRNB", f"{G['non_breakout_days']['mean_atr_pct']:.2f}", mg, "non_breakout_days.mean_atr_pct")
    M("GAgree", f(G["label_agreement_with_frozen_label_10_B0"], 1, pct=True), mg, "label_agreement_with_frozen_label_10_B0")

    # ---------------- post-hoc A (placebo) - placeholder until the 1,000-draw run has landed
    ma_ = "results/posthoc/A_placebo.json"
    if A.get("draws", 0) >= 1000:
        a = A["BROAD"]
        M("placeboDraws", f"{A['draws']:,}", ma_, "draws")
        M("placeboActual", f(a["actual_diff_pp"], 2, True), ma_, "BROAD.actual_diff_pp")
        M("placeboMean", f(a["placebo_mean_pp"], 2, True), ma_, "BROAD.placebo_mean_pp")
        M("placeboSD", f"{a['placebo_sd_pp']:.2f}", ma_, "BROAD.placebo_sd_pp")
        M("placeboP", f"{a['p_one_sided_placebo_ge_actual']:.3f}", ma_, "BROAD.p_one_sided_placebo_ge_actual")
        M("placeboPct", f"{a['percentile_of_actual']:.0f}", ma_, "BROAD.percentile_of_actual")
    else:
        for n in ["placeboDraws", "placeboActual", "placeboMean", "placeboSD", "placeboP", "placeboPct"]:
            M(n, "\\textbf{[pending]}", ma_, "PLACEHOLDER - placebo still running")
    p7p = p7["placebo"]["BROAD"]
    M("PSevenActual", f(p7p["actual_diff"], 2, True, True), "results/metrics/p7_robustness.json", "placebo.BROAD.actual_diff")
    M("PSevenDraws", len(p7p["placebo_draws"]), "results/metrics/p7_robustness.json", "len(placebo.BROAD.placebo_draws)")
    M("PSevenGE", f(p7p["share_placebo_ge_actual"], 0, pct=True), "results/metrics/p7_robustness.json", "placebo.BROAD.share_placebo_ge_actual")

    lines = ["% generated by scripts/build_macros.py from result JSON - do not edit by hand"]
    lines += [f"\\newcommand{{\\{n}}}{{{v}}}" for n, v, _, _ in MACROS]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(PAPER / "numbers_sources.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["macro", "value", "source_file", "key"])
        w.writerows(MACROS)
    print(len(MACROS), "macros written;", sum(1 for m in MACROS if "pending" in m[1]), "pending")


if __name__ == "__main__":
    main()
