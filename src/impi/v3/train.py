"""IMPI v3 model development - option-implied features + 20-day targets (pre-registered: pre_registered_v3.md).

    python -m impi.v3.train

Protocol (identical to v2; no 2023+ data in any choice):
  * rows: eligible Tier-1 stock-days from 2015-06-01 whose label window ends by 2022-12-31;
  * walk-forward CV, test folds = calendar years 2018..2022, purged by each target's own label end,
    15-calendar-day embargo;
  * stage 1: the four feature sets with XGBoost config c1;
  * stage 2: on the best stage-1 set, XGBoost c2 and c3, logistic regression, and the probability-average
    ensemble of the best XGBoost config with the logistic model;
  * chosen model per target = highest mean fold AUC; final refit on all development rows; isotonic
    calibration on the out-of-fold scores of 2021-2022;
  * adoption vs v2 (pre-registered): a v3 model replaces the v2 model of the same target in the live system
    only if its CV mean AUC >= v2's + 0.005 AND it is higher in >= 4 of the 5 folds;
  * the 2023+ period is scored ONCE at the end and reported as POST-HOC; it does not affect any choice.
Every variant is logged to results/results_log.csv (phase "V3-posthoc").
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from impi.common import METRICS, PROCESSED, log, log_result, save_json, seed
from impi.models.ml import FEATURES_V1
from impi.v2.train import DEV_END, EMBARGO, EXTRA, FOLDS, REGIME, STOCK_V1, TRAIN_START, XGB_CFG as V2_CFG
from impi.v2.train import daily_ic, make_logit, probability, score
from impi.v3.panel import OPT, PANEL_V3

MODELS = PROCESSED / "models"
V3_FILE = MODELS / "v3_models.pkl"
TARGETS = ["vol_exp", "label_xs10", "label_10", "label_xs20", "label_20"]
LABEL_END = {"label_10": "label_end", "label_xs10": "label_end", "vol_exp": "label_end",
             "label_20": "label_end_20", "label_xs20": "label_end_20"}
FWD = {"label_xs10": "fwd_10", "label_xs20": "fwd_20"}
XS_COLS = STOCK_V1 + EXTRA + OPT
FEATURE_SETS = {
    "v1_raw": FEATURES_V1,                                         # v2 baselines
    "v1v2_raw": FEATURES_V1 + EXTRA,
    "v1v2opt_raw": FEATURES_V1 + EXTRA + OPT,                      # + option-implied features
    "xs_rank_opt_regime": [f"xr_{c}" for c in XS_COLS] + REGIME,
}
XGB_CFG = dict(V2_CFG)
XGB_CFG["c3"] = dict(max_depth=6, learning_rate=0.03, n_estimators=800, min_child_weight=200, subsample=0.7,
                     colsample_bytree=0.5, reg_lambda=10.0)
ADOPT_MIN_GAIN, ADOPT_MIN_FOLDS = 0.005, 4


def add_xs_ranks(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("date")
    ranks = {f"xr_{c}": g[c].rank(pct=True).astype("float32") for c in XS_COLS}
    return pd.concat([p.drop(columns=[c for c in ranks if c in p.columns]), pd.DataFrame(ranks, index=p.index)], axis=1)


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL_V3)
    p = p[p["eligible"]].reset_index(drop=True)
    return add_xs_ranks(p)


def make_xgb(cfg: str, y: np.ndarray) -> XGBClassifier:
    pos = max(y.sum(), 1)
    spw = min((len(y) - pos) / pos, 10.0)
    return XGBClassifier(**XGB_CFG[cfg], objective="binary:logistic", eval_metric="auc", tree_method="hist",
                         scale_pos_weight=spw, random_state=seed(11), n_jobs=4, verbosity=0)


def fold_split(p: pd.DataFrame, year: int, target: str):
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    le = pd.to_datetime(p[LABEL_END[target]])
    lab = p[target].notna()
    tr = lab & (p["date"] >= TRAIN_START) & (le < start - EMBARGO)
    te = lab & (p["date"] >= start) & (p["date"] <= end) & (le <= DEV_END)
    return np.flatnonzero(tr.to_numpy()), np.flatnonzero(te.to_numpy())


def cv_variant(p, target, fset, model):
    cols = FEATURE_SETS[fset]
    oof = np.full(len(p), np.nan)
    aucs = []
    for yr in FOLDS:
        tr, te = fold_split(p, yr, target)
        X, y = p.iloc[tr][cols], p.iloc[tr][target].to_numpy().astype(int)
        m = make_xgb(model, y) if model in XGB_CFG else make_logit()
        m.fit(X, y)
        s = m.predict_proba(p.iloc[te][cols])[:, 1]
        oof[te] = s
        aucs.append(roc_auc_score(p.iloc[te][target].astype(int), s))
    res = {"target": target, "feature_set": fset, "model": model, "fold_auc": [round(a, 4) for a in aucs],
           "mean_auc": float(np.mean(aucs)), "min_auc": float(np.min(aucs))}
    log_result("V3-posthoc", f"cv_{target}", f"{fset}|{model}", "train+validation(walk-forward)",
               int((~np.isnan(oof)).sum()), "mean_auc", res["mean_auc"], res, seed_value=seed(11))
    log(f"    {target:10s} {fset:19s} {model:5s} AUC {res['mean_auc']:.4f} folds {res['fold_auc']}")
    return res, oof


def develop(p, target):
    log(f"== target {target}")
    results, oofs = [], {}
    for fset in FEATURE_SETS:
        r, o = cv_variant(p, target, fset, "c1")
        results.append(r)
        oofs[(fset, "c1")] = o
    best_set = max(results, key=lambda r: r["mean_auc"])["feature_set"]
    for model in ["c2", "c3", "logit"]:
        r, o = cv_variant(p, target, best_set, model)
        results.append(r)
        oofs[(best_set, model)] = o
    xgb_best = max((r for r in results if r["feature_set"] == best_set and r["model"] in XGB_CFG),
                   key=lambda r: r["mean_auc"])["model"]
    ens = np.nanmean(np.vstack([oofs[(best_set, xgb_best)], oofs[(best_set, "logit")]]), axis=0)
    aucs = [roc_auc_score(p.iloc[te][target].astype(int), ens[te]) for te in (fold_split(p, yr, target)[1] for yr in FOLDS)]
    er = {"target": target, "feature_set": best_set, "model": f"ensemble({xgb_best}+logit)",
          "fold_auc": [round(a, 4) for a in aucs], "mean_auc": float(np.mean(aucs)), "min_auc": float(np.min(aucs))}
    log_result("V3-posthoc", f"cv_{target}", f"{best_set}|ensemble", "train+validation(walk-forward)",
               int((~np.isnan(ens)).sum()), "mean_auc", er["mean_auc"], er, seed_value=seed(11))
    log(f"    {target:10s} {best_set:19s} ensemble AUC {er['mean_auc']:.4f} folds {er['fold_auc']}")
    results.append(er)
    oofs[(best_set, "ensemble")] = ens
    chosen = max(results, key=lambda r: r["mean_auc"])
    key = (chosen["feature_set"], "ensemble" if chosen["model"].startswith("ensemble") else chosen["model"])
    no_opt = max((r for r in results if r["feature_set"] in ("v1_raw", "v1v2_raw")), key=lambda r: r["mean_auc"])
    return {"results": results, "chosen": chosen, "xgb_best": xgb_best, "oof": oofs[key], "best_without_options": no_opt}


def fit_final(p, target, dev):
    ch = dev["chosen"]
    cols = FEATURE_SETS[ch["feature_set"]]
    lab = p[target].notna()
    tr = lab & (p["date"] >= TRAIN_START) & (pd.to_datetime(p[LABEL_END[target]]) <= DEV_END)
    X, y = p.loc[tr, cols], p.loc[tr, target].to_numpy().astype(int)
    parts = {}
    if ch["model"].startswith("ensemble"):
        parts["xgb"] = make_xgb(dev["xgb_best"], y).fit(X, y)
        parts["logit"] = make_logit().fit(X, y)
    elif ch["model"] in XGB_CFG:
        parts["xgb"] = make_xgb(ch["model"], y).fit(X, y)
    else:
        parts["logit"] = make_logit().fit(X, y)
    o = dev["oof"]
    msk = ~np.isnan(o) & (p["date"] >= pd.Timestamp("2021-01-01")).to_numpy() & lab.to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(o[msk], p.loc[msk, target].astype(int))
    return {"target": target, "cols": cols, "parts": parts, "kind": ch["model"], "iso": iso, "n_train": int(tr.sum()),
            "feature_set": ch["feature_set"], "version": "v3"}


def adoption(dev: dict) -> dict:
    v2 = json.loads((METRICS / "v2_models.json").read_text(encoding="utf-8"))["chosen"]
    out = {}
    for t, d in dev.items():
        ch = d["chosen"]
        if t not in v2:
            out[t] = {"v2_exists": False, "adopt": True, "rule": "new target (no v2 model)"}
            continue
        f3, f2 = ch["fold_auc"], v2[t]["fold_auc"]
        gain = ch["mean_auc"] - v2[t]["mean_auc"]
        wins = int(sum(a > b for a, b in zip(f3, f2)))
        out[t] = {"v2_exists": True, "v2_cv_auc": v2[t]["mean_auc"], "v3_cv_auc": ch["mean_auc"], "gain": gain,
                  "folds_better": wins, "adopt": bool(gain >= ADOPT_MIN_GAIN and wins >= ADOPT_MIN_FOLDS)}
    return out


def posthoc_test(p, models):
    """Scored ONCE. Post-hoc: 2023+ was already used for the pre-registered v1 evaluation."""
    out = {"label": "POST-HOC (2023+ previously used for the pre-registered v1 evaluation)"}
    te = p[p["date"] >= pd.Timestamp("2023-01-01")]
    for t, m in models.items():
        d = te[te[t].notna()]
        raw = score(m, d)
        y = d[t].astype(int).to_numpy()
        pr = probability(m, raw)
        r = {"n": int(len(d)), "base_rate": float(y.mean()), "auc": float(roc_auc_score(y, raw)),
             "pr_auc": float(average_precision_score(y, raw)), "brier_calibrated": float(brier_score_loss(y, pr)),
             "brier_base_rate": float(brier_score_loss(y, np.full(len(y), y.mean())))}
        dec = pd.qcut(pd.Series(raw).rank(method="first"), 10, labels=False)
        r["rate_by_decile"] = d.assign(_d=dec.to_numpy()).groupby("_d")[t].mean().round(4).tolist()
        yr = d.assign(_s=raw, _y=y).groupby(d["date"].dt.year)
        r["auc_by_year"] = {int(k): round(float(roc_auc_score(g["_y"], g["_s"])), 4) for k, g in yr if g["_y"].nunique() == 2}
        if t in FWD:
            h = int(FWD[t].split("_")[1])
            dd = d.assign(_s=raw).rename(columns={FWD[t]: "_f"})
            r["daily_ic"] = daily_ic(dd.rename(columns={"_f": "fwd_10"}), raw)
            dd["_rk"] = dd.groupby("date")["_s"].rank(pct=True)
            spread = (dd[dd["_rk"] >= 0.9].groupby("date")["_f"].mean() - dd.groupby("date")["_f"].mean()).dropna()
            nonover = spread.iloc[::h]
            r[f"top_decile_minus_all_fwd{h}_gross"] = float(nonover.mean())
            r["top_decile_spread_t"] = float(nonover.mean() / nonover.std(ddof=1) * np.sqrt(len(nonover)))
            r["top_decile_minus_all_net_of_costs"] = float(nonover.mean() - 0.0045)
            r["n_windows"] = int(len(nonover))
        out[t] = r
        log_result("V3-posthoc", f"test_{t}", m["kind"], "test(post-hoc)", r["n"], "auc", r["auc"], r)
        log(f"  post-hoc {t}: AUC {r['auc']:.4f} (n {r['n']:,}, base {r['base_rate']:.3f})")
    return out


def main():
    t0 = time.time()
    p = load_panel()
    log(f"panel loaded: {p.shape}")
    dev, models = {}, {}
    for t in TARGETS:
        dev[t] = develop(p, t)
        models[t] = fit_final(p, t, dev[t])
        log(f"  final {t}: {dev[t]['chosen']['feature_set']} | {dev[t]['chosen']['model']} "
            f"CV {dev[t]['chosen']['mean_auc']:.4f} (n_train {models[t]['n_train']:,})")
    adopt = adoption(dev)
    MODELS.mkdir(parents=True, exist_ok=True)
    raw = pickle.dumps(models)
    V3_FILE.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    log(f"v3 models frozen: sha256 {sha[:16]}; adoption {adopt}")
    test = posthoc_test(p, models)                      # after the freeze, once
    out = {"protocol": "IMPI v3 (pre_registered_v3.md): option-implied features + 20-day targets",
           "cv": {t: d["results"] for t, d in dev.items()}, "chosen": {t: d["chosen"] for t, d in dev.items()},
           "best_without_options": {t: d["best_without_options"] for t, d in dev.items()},
           "adoption": adopt, "posthoc_test": test, "model_sha256": sha, "seconds": round(time.time() - t0)}
    save_json(out, METRICS / "v3_models.json")
    log(f"v3 done in {out['seconds']}s")


if __name__ == "__main__":
    main()
