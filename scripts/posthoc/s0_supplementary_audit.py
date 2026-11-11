"""Supplementary audit of the locked IMPI v1 test-set evaluation (requested 11 Nov 2026).

READ-ONLY with respect to the study: it loads the frozen events, labels, features and models and
changes none of them. It never calls log_result()/save_json() of the pipeline, so results_log.csv,
test_lock.json and p6_test.json are untouched (their SHA-256 are printed before and after).
Output: results/posthoc/s0_supplementary_audit.json (new file) + a printed table.

Every item is labelled PRE-REGISTERED (re-report of a pre-registered number) or POST-HOC
(new analysis designed after the test set was opened on 10 Oct 2026).

    .venv/Scripts/python.exe scripts/posthoc/s0_supplementary_audit.py [--placebo-draws 1000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from impi.backtest import costs as C  # noqa: E402
from impi.backtest.executor import run_backtest  # noqa: E402
from impi.common import METRICS, RESULTS, RESULTS_LOG, config  # noqa: E402
from impi.features.delivery import delivery_features  # noqa: E402
from impi.models import ml  # noqa: E402
from impi.models.calibration import brier  # noqa: E402
from impi.pipeline.common_phase import tier1  # noqa: E402
from impi.pipeline.ladder import load_closes, signals_from  # noqa: E402
from impi.pipeline.p2_events import assert_frozen, load_events  # noqa: E402
from impi.pipeline.p4_composite import MODELS  # noqa: E402
from impi.research.hypothesis_tests import amihud_buckets, test_h1  # noqa: E402
from impi.research.stats import mde_two_proportions, month_block_bootstrap  # noqa: E402
from impi.transforms import pit_store as pit  # noqa: E402
from impi.validation.multiple_testing import benjamini_hochberg  # noqa: E402

F = ml.FEATURES_V1
MARGIN = 0.02
WATCH = [RESULTS_LOG, RESULTS / "test_lock.json", METRICS / "p6_test.json", METRICS / "p6_confirmatory_table.csv",
         MODELS / "ml_frozen.pkl", MODELS / "composite.pkl"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "missing"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def h1_h2_frames(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Exactly the exposure definitions of test_h1 / test_h2."""
    h1 = d[d["deliv_z20_prev"].notna() & d["label_10"].notna()].copy()
    h1["exposed"] = h1["deliv_z20_prev"] > 0
    h2 = d[d["ret_20"].notna() & d["oi_chg20"].notna() & d["label_10"].notna()].copy()
    h2["exposed"] = (h2["ret_20"] > 0) & (h2["oi_chg20"] < 0)
    return {"H1": h1, "H2": h2}


def tost(d: pd.DataFrame, margin: float, r) -> dict:
    a, b = d.loc[d.exposed, "label_10"].astype(float), d.loc[~d.exposed, "label_10"].astype(float)
    n1, n2 = len(a), len(b)
    if min(n1, n2) < 2:
        return {"n1": n1, "n2": n2, "note": "too few events"}
    p1, p2 = a.mean(), b.mean()
    diff = p1 - p2
    se = float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))       # unpooled (Wald) SE for TOST
    p_lower = float(1 - stats.norm.cdf((diff + margin) / se))           # H0: diff <= -margin
    p_upper = float(stats.norm.cdf((diff - margin) / se))               # H0: diff >= +margin
    ci90 = (diff - 1.6448536 * se, diff + 1.6448536 * se)
    return {"n1": n1, "n2": n2, "diff": diff, "se": se, "p_lower": p_lower, "p_upper": p_upper,
            "p_tost": max(p_lower, p_upper), "ci90_wald": list(ci90),
            "equivalent_at_alpha_0.05": bool(max(p_lower, p_upper) < 0.05),
            "small_sample_flag": bool(min(n1, n2) < 30)}


def block_ci90(d: pd.DataFrame, n: int, seed_: int) -> list[float]:
    """90% percentile CI of the rate difference, resampling calendar months (same scheme as the study)."""
    r = np.random.default_rng(seed_)
    months = d["date"].dt.to_period("M")
    groups = {m: g for m, g in d.groupby(months)}
    keys = list(groups)
    vals = []
    for _ in range(n):
        s = pd.concat([groups[keys[i]] for i in r.integers(0, len(keys), len(keys))])
        if s.exposed.any() and (~s.exposed).any():
            vals.append(s.loc[s.exposed, "label_10"].mean() - s.loc[~s.exposed, "label_10"].mean())
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def psi(train: pd.Series, test: pd.Series, bins: int = 10) -> dict:
    tr, te = train.astype(float), test.astype(float)
    edges = np.unique(np.nanquantile(tr.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(x):
        cnt = np.histogram(x.dropna(), edges)[0].astype(float)
        cnt = np.append(cnt, x.isna().sum())                           # missing values = own bin
        return np.clip(cnt / cnt.sum(), 1e-4, None)
    a, b = dist(tr), dist(te)
    return {"psi": float(np.sum((b - a) * np.log(b / a))), "n_bins_used": int(len(edges) - 1),
            "missing_train": float(tr.isna().mean()), "missing_test": float(te.isna().mean())}


def main(placebo_draws: int, placebo_budget_s: float):
    t_start = time.time()
    before = {p.name: sha(p) for p in WATCH}
    assert_frozen()
    out: dict = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
                 "note": "Supplementary audit. PRE-REGISTERED = re-report of a pre-registered quantity; "
                         "POST-HOC = designed after the test set was opened (10 Oct 2026 09:23).",
                 "checksums_before": before}
    p6 = json.loads((METRICS / "p6_test.json").read_text())
    blob = pickle.loads((MODELS / "ml_frozen.pkl").read_bytes())
    with open(MODELS / "composite.pkl", "rb") as f:
        comp = pickle.load(f)
    ev = load_events()
    cuts = amihud_buckets(tier1(ev, "broad", ["train"]))
    fams = {"STRICT": tier1(ev, "strict", ["test"]), "BROAD": tier1(ev, "broad", ["test"])}
    pilot = pd.Timestamp(config()["splits"]["pilot_window_start"])

    # ---------------------------------------------------------------- 1. BH family (PRE-REGISTERED)
    rows = []
    for label, fam in [("full", p6["confirmatory"]), ("without_pilot", p6["confirmatory_without_pilot_window"])]:
        ps = [t["p"] if t["p"] is not None else np.nan for t in fam]
        qv, rej = benjamini_hochberg(ps, float(config()["models"]["fdr_q"]))
        ok = [i for i, p in enumerate(ps) if np.isfinite(p)]
        rank = {i: k + 1 for k, i in enumerate(sorted(ok, key=lambda i: ps[i]))}
        for i, t in enumerate(fam):
            rows.append({"set": label, "test": f"{t['hypothesis']}_{t['family']}", "n": t["n"], "diff": t.get("diff"),
                         "p": ps[i], "p_unguarded": t.get("p_unguarded"), "bh_rank": rank.get(i), "m": len(ok),
                         "bh_q_recomputed": qv[i], "bh_q_locked": t.get("bh_q"), "reject_q10": bool(rej[i]),
                         "direction_ok": t.get("direction_ok"), "label": t.get("label", "ok")})
    out["1_bh_family"] = {"status": "PRE-REGISTERED", "rows": rows,
                          "note": "STRICT H1-H3 have n<30 per arm -> p withheld by the pre-registered guard, so BH "
                                  "runs over m=5 finite p-values (p_unguarded shown for information only)."}

    # ---------------------------------------------------------------- 2. TOST (POST-HOC) + cost link
    cc = config()["costs"]
    notional = cc["initial_capital"] / cc["max_positions"]
    rt = {b: C.round_trip_pct(notional, b) for b in ["large", "mid", "small"]}
    bt_broad = next(b for b in p6["backtests"] if b["variant"] == "B2b_broad")
    mix = {k: v["count"] for k, v in bt_broad["by_bucket"].items()}
    rt_w = sum(rt[k] * mix[k] for k in mix) / sum(mix.values())
    tb = fams["BROAD"]
    win_ret = float(tb.loc[tb["label_10"] == 1, "ret_tb_10"].mean())
    loss_ret = float(tb.loc[tb["label_10"] == 0, "ret_tb_10"].mean())
    spread = win_ret - loss_ret
    cost_block = {"notional_per_trade_rs": notional,
                  "round_trip_pct_by_bucket": {k: 100 * v for k, v in rt.items()},
                  "round_trip_pct_weighted_B2b_test_mix": 100 * rt_w, "bucket_mix_B2b_test": mix,
                  "mean_tb_return_success_pct": 100 * win_ret, "mean_tb_return_failure_pct": 100 * loss_ret,
                  "success_minus_failure_return_pp": 100 * spread,
                  "cost_implied_success_rate_margin_pp": 100 * rt_w / spread,
                  "explanation": "A change of d in success rate changes expected gross return per trade by "
                                 "d x (mean return of successes - mean return of failures). The success-rate change "
                                 "whose return gain just pays one round trip is cost / spread."}
    tost_rows = []
    for setname, cond in [("full", lambda d: d), ("without_pilot", lambda d: d[d["date"] < pilot])]:
        for fam, d in fams.items():
            for h, dd in h1_h2_frames(cond(d)).items():
                r_ = tost(dd, MARGIN, np.random.default_rng(611))
                r_["ci90_month_block"] = block_ci90(dd, 1000, 612) if min(r_["n1"], r_["n2"]) >= 2 else None
                tost_rows.append({"set": setname, "family": fam, "hypothesis": h, **r_})
    out["2_tost"] = {"status": "POST-HOC", "margin_pp": 100 * MARGIN, "rows": tost_rows, "cost_link": cost_block}

    # ---------------------------------------------------------------- 3. MDE (PRE-REGISTERED, reproduced)
    mde = []
    for fam, d in fams.items():
        for h, dd in h1_h2_frames(d).items():
            # pre-registration (p2_events.summarise) used the family base rate and arm sizes of all labelled events
            e = (d["deliv_z20_prev"] > 0) if h == "H1" else ((d["ret_20"] > 0) & (d["oi_chg20"] < 0))
            p = float(d["label_10"].mean())
            mde.append({"family": fam, "hypothesis": h, "n": len(d), "base_rate": p, "n_exposed": int(e.sum()),
                        "n_rest": int((~e).sum()),
                        "mde_pp_prereg_method": 100 * mde_two_proportions(p, int(e.sum()), int((~e).sum())),
                        "mde_pp_analysed_arms": 100 * mde_two_proportions(float(dd["label_10"].mean()),
                                                                          int(dd.exposed.sum()), int((~dd.exposed).sum()))})
    out["3_mde"] = {"status": "PRE-REGISTERED (formula and test-set n fixed in pre_registered_hypotheses.md)",
                    "alpha": 0.05, "power": 0.8, "rows": mde,
                    "prereg_table_pp": {"BROAD_H1": 6.26, "BROAD_H2": 5.71, "STRICT_H1": 52.02, "STRICT_H2": 44.55}}

    # ---------------------------------------------------------------- 4. H1 excl. pre-expiry window (POST-HOC)
    fo = pit.read("fo_stock", columns=["date", "near_expiry"])
    expiries = np.array(sorted(pd.to_datetime(fo["near_expiry"]).dropna().unique()), dtype="datetime64[ns]")
    from impi.features.build import OUT
    cal = np.array(sorted(pd.read_parquet(OUT / "base.parquet", columns=["date"])["date"].unique()), dtype="datetime64[ns]")
    test_exp = [x for x in expiries if pd.Timestamp("2023-01-01") <= x <= pd.Timestamp("2026-10-31")]

    def window_days(include_expiry: bool) -> set:
        s = set()
        for x in test_exp:
            k = np.searchsorted(cal, x)                     # index of the expiry session (or next session)
            lo = k - 5
            hi = k + 1 if include_expiry else k
            s.update(pd.DatetimeIndex(cal[max(lo, 0):hi]))
        return s
    h1x = []
    for variant, inc in [("5_sessions_before_expiry_day", False), ("expiry_day_plus_4_before", None),
                         ("5_sessions_before_plus_expiry_day", True)]:
        if inc is None:                                      # T-4..T(expiry): 5 sessions ending on expiry day
            s = set()
            for x in test_exp:
                k = np.searchsorted(cal, x)
                s.update(pd.DatetimeIndex(cal[max(k - 4, 0):k + 1]))
        else:
            s = window_days(inc)
        for fam, d in fams.items():
            keep = d[~d["date"].isin(s)]
            r_ = test_h1(keep, cuts, 1000, np.random.default_rng(640))
            h1x.append({"window": variant, "family": fam, "n_removed": int(len(d) - len(keep)), "n": r_["n"],
                        "p_exposed": r_.get("p1"), "p_rest": r_.get("p2"), "diff": r_["diff"], "p": r_["p"],
                        "p_unguarded": r_.get("p_unguarded"), "boot_ci": [r_["bootstrap"]["ci_low"], r_["bootstrap"]["ci_high"]],
                        "label": r_["label"]})
    out["4_h1_ex_expiry"] = {"status": "POST-HOC", "n_monthly_expiries_in_test": len(test_exp),
                             "expiry_weekday_counts": pd.Series(pd.DatetimeIndex(test_exp).day_name()).value_counts().to_dict(),
                             "expiry_source": "distinct near-month stock-futures expiry dates in the F&O bhavcopy (fo_stock.near_expiry)",
                             "rows": h1x}

    # ---------------------------------------------------------------- 5. 1-AUC + PSI (POST-HOC)
    te_b = fams["BROAD"]
    tr_b = tier1(ev, "broad", ["train"])
    y = te_b["label_10"].astype(int).to_numpy()
    s_xgb = ml.predict("xgb", blob["models"]["xgb"], te_b[F].astype(float), blob["medians"])
    auc = float(roc_auc_score(y, s_xgb))
    shap = pd.read_csv(METRICS / "p5_shap_importance.csv").sort_values("mean_abs_shap", ascending=False).head(10)
    psis = [{"feature": f, "mean_abs_shap": float(v), **psi(tr_b[f], te_b[f])} for f, v in zip(shap["feature"], shap["mean_abs_shap"])]
    out["5_auc_psi"] = {"status": "POST-HOC (AUC itself is the locked P6 number)", "xgb_test_auc": auc,
                        "xgb_test_auc_locked": p6["ml_test"]["xgb"]["auc"], "one_minus_auc": 1 - auc,
                        "auc_of_flipped_score": float(roc_auc_score(y, -s_xgb)),
                        "psi_reference": "train BROAD Tier-1 events (n=%d) vs test (n=%d); 10 train-decile bins + missing bin; "
                                         "rule of thumb <0.10 stable, 0.10-0.25 moderate, >0.25 large" % (len(tr_b), len(te_b)),
                        "psi_top10_shap": psis}

    # ---------------------------------------------------------------- 6. Brier (POST-HOC)
    cal_p = blob["calibrators"]["xgb"].predict(s_xgb)
    va_b = tier1(ev, "broad", ["validation"])
    base_test = float(y.mean())
    out["6_brier"] = {"status": "POST-HOC",
                      "test_base_rate": base_test,
                      "brier_constant_at_test_base_rate": brier(np.full(len(y), base_test), y),
                      "brier_constant_at_validation_base_rate": brier(np.full(len(y), float(va_b["label_10"].mean())), y),
                      "validation_base_rate": float(va_b["label_10"].mean()),
                      "brier_xgb_calibrated": brier(cal_p, y), "brier_xgb_calibrated_locked": p6["ml_test"]["xgb"]["brier_calibrated"],
                      "brier_xgb_raw": brier(s_xgb, y),
                      "note": "The test-base-rate constant uses the test outcome rate (an oracle no model could know in "
                              "advance); the validation-base-rate constant is the fair no-skill comparator."}

    # ---------------------------------------------------------------- 7. placebo (POST-HOC)
    out["7_placebo"] = placebo(fams, placebo_draws, placebo_budget_s)

    # ---------------------------------------------------------------- 8. incomplete label windows
    data_end = pd.Timestamp(cal[-1])
    last10 = pd.DatetimeIndex(cal[-10:])
    t1e = ev[ev["tier1"] & ev["eligible"]]
    recent = t1e[t1e["date"].isin(last10)]
    inc = []
    for fam, g in recent.groupby("family"):
        inc.append({"family": fam, "events_last10": len(g), "label_10_missing": int(g["label_10"].isna().sum()),
                    "label_10_present": int(g["label_10"].notna().sum()),
                    "label_end_after_test_end": int((pd.to_datetime(g["label_end"]) > pd.Timestamp(config()["splits"]["test"][1])).sum()),
                    "in_test_split": int((g["split"] == "test").sum())})
    all_missing = t1e[t1e["label_10"].isna()]
    out["8_incomplete_labels"] = {"status": "PRE-REGISTERED rule (labels.yaml / triple_barrier.py, frozen 08:53 10 Oct)",
                                  "data_end": str(data_end.date()), "last10_sessions": [str(x.date()) for x in last10],
                                  "test_end": config()["splits"]["test"][1], "rows": inc,
                                  "all_tier1_eligible_with_missing_label_10": all_missing.groupby("family").size().to_dict(),
                                  "handling": "triple_barrier(): if the data simply ends before T+10 the label is NaN "
                                              "(stocks that stop trading get 0 instead, D-017). assign_split() puts an "
                                              "event in a split only if label_end is known and <= split end, and tier1() "
                                              "keeps only label_10.notna(); so these events are excluded from every "
                                              "hypothesis test, model and backtest - not imputed, not counted as failures."}

    # ---------------------------------------------------------------- 9. composite / H3 / 165 vs 157
    mc = config()["models"]["composite"]
    out["9a_composite"] = {"status": "PRE-REGISTERED", "weights": mc["weights"], "tier2_weights": mc["tier2_weights"],
                           "threshold": mc["threshold"], "clip": mc["clip"], "standardise_on": mc["standardise_on"],
                           "train_mean_sd": {k: list(v) for k, v in comp.params.items()},
                           "formula": "Score = 0.30*CQ + 0.25*DZ + 0.25*PA + 0.20*VA, each component a train-split z "
                                      "clipped to [-3,3], missing -> 0; CQ = z(mean(-z bbw_pct252, -z atr_ratio, "
                                      "+z nr7_count10, -z range_contraction)); DZ = z(deliv_z20); PA = z(mean(z price_oi_align, "
                                      "(fii_fut_stk_cot-50)/28.87)); VA = z(vol_z). Tradeable if Score >= 1.0.",
                           "composite_top_decile_threshold_H4": blob["thresholds"]["composite_top_decile"]}
    out["9b_h3_formula"] = {"status": "PRE-REGISTERED",
                            "formula": "asymmetry(G) = mean(MFE_10 over G) / mean(|MAE_10| over G); statistic = "
                                       "asymmetry(regime) - asymmetry(rest); regime = fii_flow20 < 0 AND dii_flow20 > 0 "
                                       "(20-day change in cumulative FII / DII cash flow); p from a calendar-month block "
                                       "bootstrap (1,000 draws); expected sign +.",
                            "MFE_MAE": "MFE_10 = max high(T+1..T+10)/entry - 1; MAE_10 = min low(T+1..T+10)/entry - 1; entry = open(T+1)"}
    out["9c_165_vs_157"] = trades_gap(ev, blob, comp, cal)

    # ---------------------------------------------------------------- 10 + additions 1-2: git timeline
    out["10_timeline"] = timeline()

    after = {p.name: sha(p) for p in WATCH}
    out["checksums_after"] = after
    out["locked_files_unchanged"] = before == after
    out["runtime_s"] = round(time.time() - t_start, 1)
    dest = RESULTS / "posthoc" / "s0_supplementary_audit.json"
    dest.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print("runtime_s", out["runtime_s"])
    print("\nwritten:", dest, "| locked files unchanged:", out["locked_files_unchanged"])


def placebo(fams: dict, draws: int, budget_s: float) -> dict:
    """P7 placebo applied to the TEST set: deliv_pct shuffled across the event stocks within each day,
    delivery features recomputed with the frozen delivery_features(), H1 difference recomputed."""
    fcfg = config()["features"]
    syms = sorted(set(pd.concat(fams.values())["symbol"].astype(str)))
    warm = pd.Timestamp("2021-06-01")                       # >= 272 sessions before 2023-01-01 for the 20+252 windows
    eq = pit.read("equity_adj", columns=["date", "symbol", "series", "close", "deliv_pct"], symbols=syms, start=warm)
    eq = eq[eq["date"] >= warm]
    eq["symbol"] = eq["symbol"].astype(str)
    eq = eq.sort_values(["symbol", "date"]).reset_index(drop=True)
    m = (eq["series"] == "EQ").to_numpy()

    def h1_diff(d):
        d = d[d["deliv_z20_prev"].notna() & d["label_10"].notna()]
        e = d["deliv_z20_prev"] > 0
        return float(d.loc[e, "label_10"].mean() - d.loc[~e, "label_10"].mean()) if e.any() and (~e).any() else np.nan

    def one(dp: np.ndarray) -> dict:
        x = eq.copy()
        x["deliv_pct"] = dp
        z = []
        for s, g in x.groupby("symbol", sort=False):
            f = delivery_features(g.reset_index(drop=True), fcfg)
            z.append(pd.DataFrame({"symbol": s, "date": g["date"].to_numpy(), "dz": f["deliv_z20_prev"].to_numpy()}))
        z = pd.concat(z)
        res = {}
        for k, v in fams.items():
            vv = v.drop(columns=["deliv_z20_prev"]).merge(z.rename(columns={"dz": "deliv_z20_prev"}), on=["symbol", "date"], how="left")
            res[k] = h1_diff(vv)
        return res
    # sanity: unshuffled recomputation (truncated history) vs frozen deliv_z20_prev
    base = one(eq["deliv_pct"].to_numpy())
    actual = {k: h1_diff(v) for k, v in fams.items()}
    r = np.random.default_rng(20261111)
    sims = {k: [] for k in fams}
    t0 = time.time()
    dates = eq["date"].to_numpy()
    idx_eq = np.flatnonzero(m)
    by_day = pd.Series(idx_eq).groupby(dates[idx_eq]).apply(lambda s: s.to_numpy()).to_list()
    timing_draws = 3
    status, done = "POST-HOC", 0
    for i in range(draws):
        dp = eq["deliv_pct"].to_numpy().copy()
        for ix in by_day:
            dp[ix] = dp[r.permutation(ix)]
        res = one(dp)
        for k in fams:
            sims[k].append(res[k])
        done += 1
        if done == timing_draws:
            est = (time.time() - t0) / timing_draws * draws
            if est > budget_s:
                return {"status": "POST-HOC - SKIPPED", "reason": f"estimated {est/60:.1f} min for {draws} draws "
                        f"(> {budget_s/60:.0f} min budget); {timing_draws} timing draws taken, not reported as a result",
                        "seconds_per_draw": (time.time() - t0) / timing_draws, "actual_diff": actual}
    out = {"status": "POST-HOC", "draws": done, "seconds": round(time.time() - t0, 1),
           "design": "P7 placebo applied to the TEST set: delivery % shuffled across the test-event stocks within "
                     "each trading day (EQ rows, from 2021-06-01 for warm-up), delivery z-scores recomputed, H1 "
                     "exposed-minus-rest success-rate difference recomputed on the frozen events/labels.",
           "unshuffled_recompute_check": base}
    for k in fams:
        a = np.array(sims[k], float)
        a = a[np.isfinite(a)]
        out[k] = {"actual_diff": actual[k], "placebo_mean": float(a.mean()), "placebo_sd": float(a.std(ddof=1)),
                  "n_valid": int(len(a)),
                  "p_one_sided_ge_actual": float((np.sum(a >= actual[k]) + 1) / (len(a) + 1)),
                  "p_two_sided_abs": float((np.sum(np.abs(a - a.mean()) >= abs(actual[k] - a.mean())) + 1) / (len(a) + 1))}
    return out


def trades_gap(ev, blob, comp, cal) -> dict:
    ev = ev.copy()
    tmask = ev["tier1"] & ev["eligible"] & (ev["split"] == "test")
    ev["ml_p"] = np.nan
    ev.loc[tmask, "ml_p"] = ml.predict("xgb", blob["models"]["xgb"], ev.loc[tmask, F].astype(float), blob["medians"])
    br = tier1(ev, "broad", ["test"])
    br4 = br[br["ml_p"] >= blob["thresholds"]["ml_top_decile_raw"]]
    closes = load_closes(set(ev.loc[tmask, "symbol"].astype(str)))
    sig = signals_from(br4, ev["ml_p"])
    # replay the executor's entry rule to classify each signal (mirrors executor.run_backtest exactly)
    from impi.common import seed as seed_fn
    _, tr = run_backtest(sig, closes, cal, seed=seed_fn(400))
    s = sig.copy()
    reasons = {}
    s["_nan"] = s[["entry_date", "exit_date", "entry_px", "exit_px"]].isna().any(axis=1) | ~((s["entry_px"] > 0) & (s["exit_px"] > 0))
    executed = set(zip(tr["symbol"].astype(str), pd.to_datetime(tr["event_date"]))) if len(tr) else set()
    detail = []
    for _, r_ in s.iterrows():
        key = (str(r_["symbol"]), pd.Timestamp(r_["date"]))
        if key in executed:
            continue
        if r_["_nan"]:
            why = "no entry/exit price (label window incomplete or no T+1 bar)"
        else:
            d = r_["entry_date"]
            open_ = tr[(tr["entry_date"] < d) & (tr["exit_date"] >= d)]
            same_day = tr[tr["entry_date"] == d]
            if (open_["symbol"] == r_["symbol"]).any() or (same_day["symbol"] == r_["symbol"]).any():
                why = "same stock already held (one position per stock)"
            elif len(open_) + len(same_day) >= int(config()["costs"]["max_positions"]):
                why = "all 20 slots full"
            else:
                why = "other (cash/notional)"
        reasons[why] = reasons.get(why, 0) + 1
        detail.append({"symbol": key[0], "event_date": str(key[1].date()), "entry_date": str(pd.Timestamp(r_["entry_date"]).date())
                       if pd.notna(r_["entry_date"]) else None, "reason": why})
    return {"status": "PRE-REGISTERED backtest rules (executor.py); explanation is a post-hoc trace",
            "ml_top_decile_events": int(len(br4)), "trades": int(len(tr)), "not_traded": int(len(br4) - len(tr)),
            "reasons": reasons, "detail": detail,
            "rule": "executor.run_backtest: max 20 concurrent equal-weight positions, one position per stock at a "
                    "time, entry at T+1 open, exit at the triple-barrier exit; signals with no free slot are dropped "
                    "(not queued)."}


def timeline() -> dict:
    log = git("log", "--reverse", "--format=%h|%cI|%s")
    commits = [dict(zip(["hash", "committed", "subject"], l.split("|", 2))) for l in log.splitlines()]
    v1 = [c for c in commits if c["committed"] <= "2026-10-10T09:31"]
    added = {f: git("log", "--diff-filter=A", "--format=%h %cI", "--", f) for f in
             ["pre_registered_hypotheses.md", "results/metrics/feature_freeze.json", "results/metrics/model_freeze.json",
              "results/test_lock.json", "results/metrics/p6_test.json"]}
    lock = json.loads((RESULTS / "test_lock.json").read_text())
    rl = pd.read_csv(RESULTS_LOG)
    pre_test = rl[pd.to_datetime(rl["timestamp"]) < pd.Timestamp(lock["attempts"][0]["started"])]
    upto_p6 = rl[rl["phase"].isin(["P3", "P4", "P5", "P6"])]
    trials = upto_p6[["timestamp", "phase", "experiment", "variant", "split", "metric", "value", "git_commit"]].to_dict("records")
    summary = (upto_p6.groupby(["phase", "experiment", "split"]).agg(n=("timestamp", "size"), first=("timestamp", "min"),
                                                                       last=("timestamp", "max")).reset_index().to_dict("records"))
    logs = {}
    for name in ["p2.log", "p5.log", "p6.log", "p7.log", "full_pit.log"]:
        p = ROOT / "data" / "logs" / name
        if p.exists():
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            logs[name] = {"first": lines[0][:160] if lines else None, "last": lines[-1][:160] if lines else None}
    return {"status": "FACT (git / file timestamps, IST +05:30)",
            "v1_commits_in_order": v1, "files_first_committed": added, "test_lock": lock,
            "who_created_test_lock": "p6_test.main() writes results/test_lock.json at the START of the one-time run "
                                     "(09:23:35), recording git_commit = HEAD at that moment (ea49def = code + train/val "
                                     "results). The lock file itself was first committed in 359bcc7 (09:30:43) together "
                                     "with the P6 results.",
            "results_log_rows_through_P6": int(len(upto_p6)),
            "results_log_rows_before_test_run_started": int(len(pre_test)),
            "deflated_sharpe_n_trials_used": 89,
            "trial_groups": summary, "trials": trials, "phase_logs": logs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo-draws", type=int, default=1000)
    ap.add_argument("--placebo-budget-min", type=float, default=10.0)
    a = ap.parse_args()
    main(a.placebo_draws, a.placebo_budget_min * 60)
