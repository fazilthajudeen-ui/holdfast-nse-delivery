"""P6 - the ONE-TIME test-set evaluation (G4). Refuses to run if results/test_lock.json exists.

    python -m impi.pipeline.p6_test

Confirmatory family (BH q = 0.10): H1/H2/H3 on STRICT and BROAD test events, H4-AUC (DeLong, XGBoost
vs composite, BROAD) and H4-top-decile precision. Plus: benchmark ladders B0-B4, ML metrics and
calibration, top-decile return spread CI, costed backtests, deflated Sharpe, Hansen SPA / White RC,
with/without the Delivery Lens pilot window, within-test sub-periods, Tier-2 replication (secondary)
and an expanding-window monthly walk-forward with frozen hyper-parameters.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from impi.backtest.metrics import deflated_sharpe, portfolio_metrics, spa_tests
from impi.common import METRICS, RESULTS, RESULTS_LOG, config, config_hash, git_commit, log, log_result, rng, save_json, seed
from impi.models import ml
from impi.models.calibration import brier, reliability
from impi.pipeline.common_phase import assert_preregistered, tier1, tier2_only
from impi.pipeline.ladder import backtest_variant, ladder, load_closes
from impi.pipeline.p2_events import assert_frozen, load_events
from impi.pipeline.p4_composite import MODELS
from impi.research.hypothesis_tests import amihud_buckets, run_family, test_h1, test_h3
from impi.research.stats import delong_test, month_block_bootstrap, two_proportion_z
from impi.transforms import pit_store as pit
from impi.validation.multiple_testing import benjamini_hochberg

LOCK = RESULTS / "test_lock.json"
F = ml.FEATURES_V1


def _ml_scores(blob, d):
    X = d[F].astype(float)
    return {k: ml.predict(k, blob["models"][k], X, blob["medians"]) for k in ["xgb", "rf", "logit"]}


def h4_tests(d: pd.DataFrame, blob, comp, r, n_boot) -> list[dict]:
    y = d["label_10"].astype(int).to_numpy()
    s_ml = ml.predict("xgb", blob["models"]["xgb"], d[F].astype(float), blob["medians"])
    s_c = comp.score(d)["score"].to_numpy()
    dl = delong_test(y, s_ml, s_c)
    top_ml = s_ml >= blob["thresholds"]["ml_top_decile_raw"]
    top_c = s_c >= blob["thresholds"]["composite_top_decile"]
    dd = d.assign(_ml=top_ml, _c=top_c)
    prec = lambda x, col: float(x.loc[x[col], "label_10"].mean()) if x[col].any() else np.nan
    bt = month_block_bootstrap(dd, lambda x: prec(x, "_ml") - prec(x, "_c"), n_boot, r)
    spread = month_block_bootstrap(dd, lambda x: float(x.loc[x["_ml"], "ret_tb_10"].mean() - x.loc[x["_c"], "ret_tb_10"].mean())
                                   if x["_ml"].any() and x["_c"].any() else np.nan, n_boot, r)
    small = min(int(top_ml.sum()), int(top_c.sum())) < 30
    if small:
        bt = {**bt, "p_unguarded": bt["p"], "p": np.nan}
    return [{"hypothesis": "H4_auc", "family": "BROAD", "test": "delong", **dl, "p": dl["p"],
             "expected_sign": "+", "direction_ok": bool(dl["diff"] > 0) if np.isfinite(dl["diff"]) else None, "n": len(d)},
            {"hypothesis": "H4_top_decile", "family": "BROAD", "test": "month_block_bootstrap_precision",
             "ml_precision": prec(dd, "_ml"), "composite_precision": prec(dd, "_c"), "n_ml": int(top_ml.sum()),
             "n_comp": int(top_c.sum()), "diff": bt["estimate"], "p": bt["p"], "ci_low": bt["ci_low"],
             "ci_high": bt["ci_high"], "expected_sign": "+",
             "direction_ok": bool(bt["estimate"] > 0) if np.isfinite(bt["estimate"]) else None, "n": len(d),
             "return_spread_ml_minus_composite": spread}]


def confirmatory(ev, blob, comp, cuts, label: str) -> list[dict]:
    q = float(config()["models"]["fdr_q"])
    n_boot = int(config()["models"]["bootstrap"]["n"])
    fams = {"STRICT": tier1(ev, "strict", ["test"]), "BROAD": tier1(ev, "broad", ["test"])}
    tests = run_family(fams, cuts, rng(600))
    tests += h4_tests(fams["BROAD"], blob, comp, rng(601), n_boot)
    qv, rej = benjamini_hochberg([t["p"] for t in tests], q)
    for t, qq, rr in zip(tests, qv, rej):
        t["bh_q"], t["bh_reject"] = qq, bool(rr)
        t["supported"] = bool(rr and t.get("direction_ok"))
        log_result("P6", f"confirmatory_{label}", f"{t['hypothesis']}_{t['family']}", "test", t["n"], "p", t["p"],
                   {"diff": t.get("diff"), "bh_q": qq}, seed_value=seed(600))
    return tests


def walk_forward(ev: pd.DataFrame, blob) -> dict:
    """Expanding window, monthly refits with the frozen hyper-parameters (no re-tuning)."""
    allb = ev[(ev["family"] == "broad") & ev["tier1"] & ev["eligible"] & ev["label_10"].notna()].sort_values("date")
    test = allb[allb["split"] == "test"]
    params = dict(blob["best_params"]["xgb"])
    n_est = int(blob["models"]["xgb"].best_iteration) + 1
    preds = []
    for month, g in test.groupby(test["date"].dt.to_period("M")):
        start = month.to_timestamp()
        trn = allb[(allb["date"] >= pd.Timestamp(config()["splits"]["train"][0])) & (pd.to_datetime(allb["label_end"]) < start)]
        y = trn["label_10"].astype(int).to_numpy()
        m = ml.make_model("xgb", params, ml.class_weight(y), n_estimators=n_est)
        m.fit(trn[F].astype(float), y)
        preds.append(pd.DataFrame({"idx": g.index, "p": m.predict_proba(g[F].astype(float))[:, 1]}))
    p = pd.concat(preds).set_index("idx")["p"]
    t = test.loc[p.index]
    return {"months": len(preds), "oos_auc": float(roc_auc_score(t["label_10"].astype(int), p)),
            "oos_pr_auc": float(average_precision_score(t["label_10"].astype(int), p)), "n": len(t)}, p


def main(force: bool = False):
    assert_preregistered()
    assert_frozen()
    raw = (MODELS / "ml_frozen.pkl").read_bytes()
    frozen = json.loads((METRICS / "model_freeze.json").read_text())["sha256"]
    if hashlib.sha256(raw).hexdigest() != frozen:
        raise SystemExit("Frozen ML models changed after P5 - not allowed.")
    attempts = []
    if LOCK.exists():
        prev = json.loads(LOCK.read_text())
        attempts = prev.get("attempts", [prev])
        crashed_clean = all(a.get("status") == "crashed_no_results" for a in attempts) and not (METRICS / "p6_test.json").exists()
        if not (force or crashed_clean):
            raise SystemExit(f"Test set already evaluated ({str(prev)[:300]}). The test set is touched exactly once.")
    attempt = {"started": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
               "config_hash": config_hash(), "forced_rerun": force, "status": "running"}
    save_json({"attempts": attempts + [attempt]}, LOCK)
    try:
        _run(raw, attempts, attempt)
    except Exception:
        if not (METRICS / "p6_test.json").exists():
            attempt["status"] = "crashed_no_results"
            save_json({"attempts": attempts + [attempt]}, LOCK)
        raise


def _run(raw: bytes, attempts: list, attempt: dict):
    blob = pickle.loads(raw)
    with open(MODELS / "composite.pkl", "rb") as f:
        comp = pickle.load(f)
    ev = load_events()
    cuts = amihud_buckets(tier1(ev, "broad", ["train"]))
    n_boot = int(config()["models"]["bootstrap"]["n"])
    out = {"attempt": attempt, "previous_attempts": attempts}
    out["confirmatory"] = confirmatory(ev, blob, comp, cuts, "full")
    pilot = pd.Timestamp(config()["splits"]["pilot_window_start"])
    out["confirmatory_without_pilot_window"] = confirmatory(ev[ev["date"] < pilot], blob, comp, cuts, "no_pilot")
    # ML metrics on test
    te_b = tier1(ev, "broad", ["test"])
    y = te_b["label_10"].astype(int).to_numpy()
    sc = _ml_scores(blob, te_b)
    out["ml_test"] = {}
    for k, s in sc.items():
        cal_p = blob["calibrators"][k].predict(s)
        out["ml_test"][k] = {"auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
                             "brier_calibrated": brier(cal_p, y), "brier_raw": brier(s, y)}
    out["ml_test"]["composite_auc"] = float(roc_auc_score(y, comp.score(te_b)["score"]))
    out["ml_test"]["base_rate"] = float(y.mean())
    out["reliability_test_xgb"] = reliability(blob["calibrators"]["xgb"].predict(sc["xgb"]), y).to_dict("records")
    # ladders + backtests
    ev = ev.copy()
    ev["comp_score"] = comp.score(ev)["score"]
    ev["comp_tradeable"] = ev["comp_score"] >= comp.threshold
    tmask = ev["tier1"] & ev["eligible"] & (ev["split"] == "test")
    ev["ml_p"] = np.nan
    ev.loc[tmask, "ml_p"] = ml.predict("xgb", blob["models"]["xgb"], ev.loc[tmask, F].astype(float), blob["medians"])
    ev["ml_top"] = ev["ml_p"] >= blob["thresholds"]["ml_top_decile_raw"]
    from impi.features.build import OUT
    from impi.validation.splits import assign_split
    base = pd.read_parquet(OUT / "base.parquet")
    base["split"] = assign_split(base["date"], base["label_end"])
    b0 = base[base["eligible"] & (base["split"] == "test") & base["label_10"].notna()]
    b1, b2, br = tier1(ev, "all", ["test"]), tier1(ev, "strict", ["test"]), tier1(ev, "broad", ["test"])
    b3, b4 = b2[b2["comp_tradeable"]], b2[b2["ml_top"]]
    br3, br4 = br[br["comp_tradeable"]], br[br["ml_top"]]
    r = rng(602)
    out["ladder_strict"] = ladder([("B0_base", b0), ("B1_all_breakouts", b1), ("B2_strict", b2),
                                   ("B3_strict_composite", b3), ("B4_strict_ml_top_decile", b4)], r, n_boot // 2)
    out["ladder_broad"] = ladder([("B0_base", b0), ("B1_all_breakouts", b1), ("B2b_broad", br),
                                  ("B3b_broad_composite", br3), ("B4b_broad_ml_top_decile", br4)], r, n_boot // 2)
    cal = np.array(sorted(base["date"].unique()), dtype="datetime64[ns]")
    closes = load_closes(set(ev.loc[tmask, "symbol"].astype(str)))
    bts, rets = [], {}
    for name, d in [("B1_all_breakouts", b1), ("B2_strict", b2), ("B3_strict_composite", b3),
                    ("B4_strict_ml_top_decile", b4), ("B2b_broad", br), ("B3b_broad_composite", br3),
                    ("B4b_broad_ml_top_decile", br4)]:
        if len(d) == 0:
            continue
        prio = ev["ml_p"] if "ml" in name else ev["comp_score"]
        m_, rr = backtest_variant(name, d, closes, cal, "test", "P6", priority=prio)
        bts.append(m_)
        rets[name] = rr
    idx = pit.read("index")
    n500 = idx[idx["index"] == "nifty500"].set_index("date")["close"].sort_index()
    bench = n500.pct_change()
    vr = pd.DataFrame(rets)
    bench = bench.reindex(vr.index)
    out["benchmark_nifty500_test"] = portfolio_metrics(bench.dropna(), float(config()["costs"]["risk_free_annual"]))
    out["backtests"] = bts
    out["spa"] = spa_tests(bench, vr, int(config()["models"]["spa"]["reps"]), int(config()["models"]["spa"]["block_size"]), seed(603))
    log_df = pd.read_csv(RESULTS_LOG)
    n_trials = int(len(log_df))
    bt_rows = log_df[log_df["experiment"] == "backtest"]
    sr_daily = pd.to_numeric(bt_rows["value"], errors="coerce").dropna() / np.sqrt(252)
    var_sr = float(sr_daily.var(ddof=1)) if len(sr_daily) > 1 else np.nan
    out["deflated_sharpe"] = {}
    for m_ in bts:
        if m_.get("sharpe_daily") is not None and np.isfinite(m_.get("sharpe_daily", np.nan)):
            out["deflated_sharpe"][m_["variant"]] = {
                **deflated_sharpe(m_["sharpe_daily"], m_["days"], m_["skew"], m_["kurtosis"], var_sr, n_trials),
                "n_trials_results_log": n_trials, "var_trial_sr_daily": var_sr}
    # within-test sub-periods (structural breaks: T+1 settlement 2023, F&O changes 2024-25)
    sub = {}
    for fam, d in [("strict", b2), ("broad", br)]:
        for yr, g in d.groupby(d["date"].dt.year):
            sub[f"{fam}_{yr}"] = {"n": len(g), "success_rate": float(g["label_10"].mean()),
                                  "h1_diff": test_h1(g, cuts, 200, rng(604))["diff"] if len(g) > 30 else None}
    out["test_subperiods"] = sub
    # Tier-2 replication (secondary): delivery + compression outside the F&O segment
    t2 = {}
    for fam in ["strict", "broad"]:
        d = tier2_only(ev, fam, ["test"])
        if len(d) > 30:
            t2[fam] = test_h1(d, cuts, n_boot, rng(605))
    out["tier2_replication_H1"] = t2
    # FII and DII never only pooled (spec section 12): H3 with each side alone (secondary, outside the BH family)
    out["h3_fii_vs_dii"] = {f"{fam}_{w}": test_h3(d, n_boot, rng(606), which=w)
                            for fam, d in [("STRICT", b2), ("BROAD", br)] for w in ["fii_only", "dii_only"]}
    wf, _ = walk_forward(ev, blob)
    out["walk_forward"] = wf
    out["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    save_json(out, METRICS / "p6_test.json")
    attempt["status"] = "finished"
    attempt["finished"] = out["finished"]
    save_json({"attempts": attempts + [attempt]}, LOCK)
    tab = pd.DataFrame([{k: t.get(k) for k in ["hypothesis", "family", "n", "diff", "p", "bh_q", "expected_sign",
                                               "direction_ok", "supported"]} for t in out["confirmatory"]])
    tab.to_csv(METRICS / "p6_confirmatory_table.csv", index=False)
    log("P6 confirmatory:\n" + tab.to_string())


if __name__ == "__main__":
    import sys
    main(force="--force-rerun-i-understand" in sys.argv)
