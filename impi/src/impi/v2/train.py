"""IMPI v2 model development - POST-HOC (after the one-time v1 test), on train + validation only.

    python -m impi.v2.train

Protocol (no test data in any choice):
  * rows: eligible Tier-1 stock-days 2015-06-01 .. 2022-12-31 (label window inside that range);
  * walk-forward CV: test folds = calendar years 2018..2022; each fold trains on rows whose label
    window ended before the fold starts (purge) with a 10-trading-day embargo;
  * stage 1: feature sets compared with one XGBoost config; stage 2: second config, logistic
    regression and a probability-average ensemble on the best feature set; selection = mean fold AUC;
  * final model per target: refit on all train+validation rows; isotonic calibration on the
    out-of-fold predictions of the last two folds;
  * the test period (2023+) is scored ONCE at the end and reported as POST-HOC (the test set was
    already used for the pre-registered v1 evaluation).
Every variant is logged to results/results_log.csv (phase "V2-posthoc").
"""
from __future__ import annotations

import hashlib
import pickle
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from impi.common import METRICS, PROCESSED, log, log_result, save_json, seed
from impi.models.ml import FEATURES_V1
from impi.v2.panel import PANEL

MODELS = PROCESSED / "models"
V2_FILE = MODELS / "v2_models.pkl"
TARGETS = ["label_10", "label_xs10", "vol_exp"]
EXTRA = ["ret_1", "ret_5", "ret_60", "ret_120", "dist_52w_high", "dist_52w_low", "rv20", "rv_5_20", "rv_20_60",
         "rv20_pct252", "deliv_z5", "deliv_trend_5_20", "deliv_pct_rank252", "turnover_z20", "clv5", "gap_1", "up_days_20"]
_P = ["fii", "dii", "pro", "client"]
MARKET = [c for c in FEATURES_V1 if c.split("_")[0] in _P] + \
    ["nifty_above_200", "nifty_trend_50_200", "vix_pct252", "breadth10", "fii_flow20", "dii_flow20"]
STOCK_V1 = [c for c in FEATURES_V1 if c not in MARKET]
REGIME = ["vix_pct252", "nifty_trend_50_200", "breadth10", "fii_flow20", "dii_flow20", "fii_fut_idx_cot"]
FEATURE_SETS = {
    "v1_raw": FEATURES_V1,
    "v1v2_raw": FEATURES_V1 + EXTRA,
    "xs_rank": [f"xr_{c}" for c in STOCK_V1 + EXTRA],
    "xs_rank_regime": [f"xr_{c}" for c in STOCK_V1 + EXTRA] + REGIME,
}
XGB_CFG = {
    "c1": dict(max_depth=3, learning_rate=0.05, n_estimators=400, min_child_weight=100, subsample=0.7,
               colsample_bytree=0.6, reg_lambda=5.0),
    "c2": dict(max_depth=5, learning_rate=0.03, n_estimators=600, min_child_weight=300, subsample=0.7,
               colsample_bytree=0.5, reg_lambda=10.0),
}
FOLDS = [2018, 2019, 2020, 2021, 2022]
TRAIN_START = pd.Timestamp("2015-06-01")
DEV_END = pd.Timestamp("2022-12-31")
EMBARGO = pd.Timedelta(days=15)                      # ~10 trading days


def add_xs_ranks(p: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank of each stock-level feature among that day's Tier-1 stocks
    (uses only same-day data -> no look-ahead; removes slow market-wide drift)."""
    g = p.groupby("date")
    for c in STOCK_V1 + EXTRA:
        p[f"xr_{c}"] = g[c].rank(pct=True).astype("float32")
    return p


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL)
    missing = [c for c in set(FEATURES_V1 + EXTRA) if c not in p.columns]
    if missing:
        raise SystemExit(f"panel is missing feature columns: {sorted(missing)}")
    p = p[p["eligible"]].reset_index(drop=True)
    return add_xs_ranks(p)


def make_xgb(cfg: str, y: np.ndarray) -> XGBClassifier:
    pos = max(y.sum(), 1)
    spw = min((len(y) - pos) / pos, 10.0)
    return XGBClassifier(**XGB_CFG[cfg], objective="binary:logistic", eval_metric="auc", tree_method="hist",
                         scale_pos_weight=spw, random_state=seed(11), n_jobs=4, verbosity=0)


def make_logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))


def daily_ic(df: pd.DataFrame, score: np.ndarray) -> float:
    d = df.assign(_s=score)[["date", "_s", "fwd_10"]].dropna()
    ics = d.groupby("date").apply(lambda g: stats.spearmanr(g["_s"], g["fwd_10"])[0] if len(g) > 10 else np.nan,
                                  include_groups=False)
    return float(np.nanmean(ics))


def fold_split(p: pd.DataFrame, year: int, target: str):
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    lab = p[target].notna()
    tr = lab & (p["date"] >= TRAIN_START) & (pd.to_datetime(p["label_end"]) < start - EMBARGO)
    te = lab & (p["date"] >= start) & (p["date"] <= end) & (pd.to_datetime(p["label_end"]) <= DEV_END)
    return np.flatnonzero(tr.to_numpy()), np.flatnonzero(te.to_numpy())


def cv_variant(p: pd.DataFrame, target: str, fset: str, model: str) -> tuple[dict, np.ndarray]:
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
    msk = ~np.isnan(oof)
    if target == "label_xs10":
        res["oof_daily_ic"] = daily_ic(p[msk], oof[msk])
    log_result("V2-posthoc", f"cv_{target}", f"{fset}|{model}", "train+validation(walk-forward)", int(msk.sum()),
               "mean_auc", res["mean_auc"], res, seed_value=seed(11))
    log(f"    {target:10s} {fset:15s} {model:5s} AUC {res['mean_auc']:.4f} folds {res['fold_auc']}")
    return res, oof


def develop(p: pd.DataFrame, target: str) -> dict:
    log(f"== target {target}")
    results, oofs = [], {}
    for fset in FEATURE_SETS:
        r, o = cv_variant(p, target, fset, "c1")
        results.append(r)
        oofs[(fset, "c1")] = o
    best_set = max((r for r in results), key=lambda r: r["mean_auc"])["feature_set"]
    for model in ["c2", "logit"]:
        r, o = cv_variant(p, target, best_set, model)
        results.append(r)
        oofs[(best_set, model)] = o
    # probability-average ensemble of the best XGB config and the logistic model (row-wise, so a single
    # live row scores exactly as it would in a batch)
    xgb_best = max((r for r in results if r["feature_set"] == best_set and r["model"] in XGB_CFG),
                   key=lambda r: r["mean_auc"])["model"]
    ens = np.nanmean(np.vstack([oofs[(best_set, xgb_best)], oofs[(best_set, "logit")]]), axis=0)
    ens_aucs = []
    for yr in FOLDS:
        _, te = fold_split(p, yr, target)
        ens_aucs.append(roc_auc_score(p.iloc[te][target].astype(int), ens[te]))
    er = {"target": target, "feature_set": best_set, "model": f"ensemble({xgb_best}+logit)",
          "fold_auc": [round(a, 4) for a in ens_aucs], "mean_auc": float(np.mean(ens_aucs)), "min_auc": float(np.min(ens_aucs))}
    log_result("V2-posthoc", f"cv_{target}", f"{best_set}|ensemble", "train+validation(walk-forward)",
               int((~np.isnan(ens)).sum()), "mean_auc", er["mean_auc"], er, seed_value=seed(11))
    log(f"    {target:10s} {best_set:15s} ensemble AUC {er['mean_auc']:.4f} folds {er['fold_auc']}")
    results.append(er)
    oofs[(best_set, "ensemble")] = ens
    chosen = max(results, key=lambda r: r["mean_auc"])
    key = (chosen["feature_set"], "ensemble" if chosen["model"].startswith("ensemble") else chosen["model"])
    return {"results": results, "chosen": chosen, "xgb_best": xgb_best, "best_set": best_set, "oof": oofs[key]}


def fit_final(p: pd.DataFrame, target: str, dev: dict):
    ch = dev["chosen"]
    cols = FEATURE_SETS[ch["feature_set"]]
    lab = p[target].notna()
    tr = lab & (p["date"] >= TRAIN_START) & (pd.to_datetime(p["label_end"]) <= DEV_END)
    X, y = p.loc[tr, cols], p.loc[tr, target].to_numpy().astype(int)
    parts = {}
    if ch["model"].startswith("ensemble"):
        parts["xgb"] = make_xgb(dev["xgb_best"], y).fit(X, y)
        parts["logit"] = make_logit().fit(X, y)
    elif ch["model"] in XGB_CFG:
        parts["xgb"] = make_xgb(ch["model"], y).fit(X, y)
    else:
        parts["logit"] = make_logit().fit(X, y)
    # isotonic calibration on out-of-fold scores of the last two CV folds (2021-2022)
    o = dev["oof"]
    msk = ~np.isnan(o) & (p["date"] >= pd.Timestamp("2021-01-01")).to_numpy() & lab.to_numpy()
    raw_oof = o[msk]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_oof, p.loc[msk, target].astype(int))
    return {"target": target, "cols": cols, "parts": parts, "kind": ch["model"], "iso": iso,
            "n_train": int(tr.sum())}


def score(model: dict, X: pd.DataFrame) -> np.ndarray:
    """Raw score: XGB / logit probability, their average for the ensemble, or a ranker's score (row-wise)."""
    X = X[model["cols"]]
    return np.mean([m.predict(X) if name == "ranker" else m.predict_proba(X)[:, 1]
                    for name, m in model["parts"].items()], axis=0)


def probability(model: dict, raw: np.ndarray) -> np.ndarray:
    return model["iso"].predict(raw)


def posthoc_test(p: pd.DataFrame, models: dict) -> dict:
    """Scored ONCE. Post-hoc: the 2023+ period was already used for the pre-registered v1 test."""
    out = {"label": "POST-HOC (test period previously used for the pre-registered v1 evaluation)"}
    te = p[(p["date"] >= pd.Timestamp("2023-01-01"))].copy()
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
        if t == "label_xs10":
            r["daily_ic"] = daily_ic(d, raw)
            dd = d.assign(_s=raw)
            dd["_rk"] = dd.groupby("date")["_s"].rank(pct=True)
            top = dd[dd["_rk"] >= 0.9].groupby("date")["fwd_10"].mean()
            allm = dd.groupby("date")["fwd_10"].mean()
            spread = (top - allm).dropna()
            nonover = spread.iloc[::10]                       # non-overlapping 10-day windows
            r["top_decile_minus_all_fwd10_gross"] = float(nonover.mean())
            r["top_decile_spread_t"] = float(nonover.mean() / nonover.std(ddof=1) * np.sqrt(len(nonover)))
            r["top_decile_minus_all_net_of_costs"] = float(nonover.mean() - 0.0045)
            r["n_windows"] = int(len(nonover))
        out[t] = r
        log_result("V2-posthoc", f"test_{t}", m["kind"], "test(post-hoc)", r["n"], "auc", r["auc"], r)
    # BROAD events with the breakout model vs the v1 numbers (XGB 0.460, composite 0.526)
    ev = pd.read_parquet(PROCESSED / "features" / "events.parquet", columns=["date", "symbol", "family", "tier1", "eligible", "label_10"])
    ev = ev[(ev["family"] == "broad") & ev["tier1"] & ev["eligible"] & ev["label_10"].notna() & (ev["date"] >= pd.Timestamp("2023-01-01"))]
    j = ev.merge(te.drop(columns=["label_10"]), on=["date", "symbol"], how="inner")
    if len(j) > 50:
        out["broad_events_label_10_auc"] = {"n": int(len(j)), "auc": float(roc_auc_score(j["label_10"].astype(int), score(models["label_10"], j))),
                                            "v1_xgb_auc": 0.460, "v1_composite_auc": 0.526}
    return out


def main():
    t0 = time.time()
    p = load_panel()
    log(f"panel loaded: {p.shape}")
    dev, models = {}, {}
    for t in TARGETS:
        dev[t] = develop(p, t)
        models[t] = fit_final(p, t, dev[t])
        log(f"  final {t}: {dev[t]['chosen']['feature_set']} | {dev[t]['chosen']['model']} (n_train {models[t]['n_train']:,})")
    MODELS.mkdir(parents=True, exist_ok=True)
    raw = pickle.dumps(models)
    V2_FILE.write_bytes(raw)
    test = posthoc_test(p, models)
    out = {"protocol": __doc__.strip().splitlines()[0], "cv": {t: d["results"] for t, d in dev.items()},
           "chosen": {t: d["chosen"] for t, d in dev.items()}, "posthoc_test": test,
           "model_sha256": hashlib.sha256(raw).hexdigest(), "seconds": round(time.time() - t0)}
    save_json(out, METRICS / "v2_models.json")
    log(f"v2 done in {out['seconds']}s: " + "; ".join(f"{t} CV {d['chosen']['mean_auc']:.3f} test {test[t]['auc']:.3f}"
                                                   for t, d in dev.items()))


if __name__ == "__main__":
    main()
