"""Extra v2 experiment (post-hoc, CV only): XGBoost learning-to-rank for the market-neutral target.

    python -m impi.v2.ranker

Pairwise ranking within each trading day (qid = date) on the quintile of the 10-day forward return,
evaluated on the SAME walk-forward folds as impi.v2.train (AUC vs label_xs10 and daily rank IC).
Adopted for the live 'rank' model only if its mean CV AUC beats the chosen classifier/ensemble.
Every variant is logged (G5).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBRanker

from impi.common import METRICS, log, log_result, save_json, seed
from impi.v2.train import (DEV_END, FEATURE_SETS, FOLDS, TRAIN_START, V2_FILE, daily_ic, fold_split, load_panel)

CFG = {"r1": dict(max_depth=4, learning_rate=0.05, n_estimators=400, min_child_weight=100, subsample=0.7,
                  colsample_bytree=0.6, reg_lambda=5.0),
       "r2": dict(max_depth=6, learning_rate=0.03, n_estimators=600, min_child_weight=300, subsample=0.7,
                  colsample_bytree=0.5, reg_lambda=10.0)}


def relevance(df: pd.DataFrame) -> np.ndarray:
    q = df.groupby("date")["fwd_10"].rank(pct=True)
    return np.clip((q * 5).fillna(0).astype(int), 0, 4).to_numpy()


def fit_ranker(df: pd.DataFrame, cols, cfg: str) -> XGBRanker:
    d = df.sort_values("date")
    qid = pd.factorize(d["date"])[0]
    m = XGBRanker(**CFG[cfg], objective="rank:pairwise", tree_method="hist", random_state=seed(21), n_jobs=4)
    m.fit(d[cols], relevance(d), qid=qid)
    return m


def main():
    p = load_panel()
    p = p[p["fwd_10"].notna()].reset_index(drop=True)
    target = "label_xs10"
    res, best = [], None
    for fset in ["v1_raw", "v1v2_raw"]:
        cols = FEATURE_SETS[fset]
        for cfg in CFG:
            oof = np.full(len(p), np.nan)
            aucs = []
            for yr in FOLDS:
                tr, te = fold_split(p, yr, target)
                m = fit_ranker(p.iloc[tr], cols, cfg)
                s = m.predict(p.iloc[te][cols])
                oof[te] = s
                aucs.append(roc_auc_score(p.iloc[te][target].astype(int), s))
            msk = ~np.isnan(oof)
            r = {"target": target, "feature_set": fset, "model": f"ranker_{cfg}", "fold_auc": [round(a, 4) for a in aucs],
                 "mean_auc": float(np.mean(aucs)), "oof_daily_ic": daily_ic(p[msk], oof[msk])}
            log_result("V2-posthoc", "cv_label_xs10_ranker", f"{fset}|ranker_{cfg}", "train+validation(walk-forward)",
                       int(msk.sum()), "mean_auc", r["mean_auc"], r, seed_value=seed(21))
            log(f"    ranker {fset:9s} {cfg} AUC {r['mean_auc']:.4f} IC {r['oof_daily_ic']:.4f} folds {r['fold_auc']}")
            res.append(r)
            if best is None or r["mean_auc"] > best[0]["mean_auc"]:
                best = (r, oof, cols, cfg)
    v2 = json.loads((METRICS / "v2_models.json").read_text())
    current = v2["chosen"][target]["mean_auc"]
    out = {"results": res, "current_xs_model_cv_auc": current, "adopted": False}
    r, oof, cols, cfg = best
    if r["mean_auc"] > current:
        lab = p[target].notna()
        tr = lab & (p["date"] >= TRAIN_START) & (pd.to_datetime(p["label_end"]) <= DEV_END)
        model = fit_ranker(p[tr], cols, cfg)
        msk = ~np.isnan(oof) & (p["date"] >= pd.Timestamp("2021-01-01")).to_numpy()
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[msk], p.loc[msk, target].astype(int))
        models = pickle.loads(V2_FILE.read_bytes())
        models[target] = {"target": target, "cols": cols, "parts": {"ranker": model}, "kind": r["model"], "iso": iso,
                          "n_train": int(tr.sum())}
        raw = pickle.dumps(models)
        V2_FILE.write_bytes(raw)
        out["adopted"] = True
        out["adopted_model"] = r
        log(f"adopted ranker for {target}: CV AUC {r['mean_auc']:.4f} > {current:.4f}")
        # keep the reports consistent: post-hoc numbers are recomputed for the adopted model
        import hashlib
        from impi.v2.train import posthoc_test
        v2["chosen"][target] = r
        v2["cv"][target].append(r)
        v2["posthoc_test"] = posthoc_test(load_panel(), models)
        v2["model_sha256"] = hashlib.sha256(raw).hexdigest()
        save_json(v2, METRICS / "v2_models.json")
    else:
        log(f"ranker not adopted: best CV AUC {r['mean_auc']:.4f} <= current {current:.4f}")
    save_json(out, METRICS / "v2_ranker.json")


if __name__ == "__main__":
    main()
