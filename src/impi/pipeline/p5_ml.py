"""P5 - ML challenger on TRAIN/VALIDATION: search (purged CV), final fit, isotonic calibration,
SHAP interactions, provisional H4 on validation. Freezes the models for the one-time test (P6).

    python -m impi.pipeline.p5_ml
"""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from impi.common import FIGURES, METRICS, config, log, log_result, save_json, seed
from impi.models import ml
from impi.models.calibration import brier, fit_isotonic, reliability
from impi.models.shap_analysis import dependence_plot, global_importance, interaction_ranking
from impi.pipeline.common_phase import assert_preregistered, tier1
from impi.pipeline.p2_events import assert_frozen, load_events
from impi.pipeline.p4_composite import MODELS
from impi.research.stats import delong_test, two_proportion_z
from impi.validation.splits import purged_kfold

F = ml.FEATURES_V1


def top_decile_precision(y, score, thr) -> tuple[float, int]:
    m = score >= thr
    return (float(y[m].mean()) if m.any() else np.nan), int(m.sum())


def main():
    assert_preregistered()
    assert_frozen()
    with open(MODELS / "composite.pkl", "rb") as f:
        comp = pickle.load(f)
    ev = load_events()
    tr = tier1(ev, "broad", ["train"]).sort_values("date")
    va = tier1(ev, "broad", ["validation"]).sort_values("date")
    tv = pd.concat([tr, va]).sort_values("date").reset_index(drop=True)
    cal = np.array(sorted(ev["date"].unique()), dtype="datetime64[ns]")
    c = config()
    folds = purged_kfold(tv["date"], tv["label_end"], c["splits"]["cv_folds"], c["splits"]["cv_embargo_days"], cal)
    Xtv, ytv = tv[F].astype(float), tv["label_10"].to_numpy().astype(int)
    best, search = {}, []
    for kind in ["xgb", "rf", "logit"]:
        p, res = ml.search(kind, Xtv, ytv, folds)
        best[kind] = p
        search.append(res)
        log(f"  {kind}: best cv PR-AUC {res['cv_pr_auc'].max():.4f} params {p}")
    pd.concat(search).to_csv(METRICS / "p5_search.csv", index=False)
    Xtr, ytr = tr[F].astype(float), tr["label_10"].to_numpy().astype(int)
    Xva, yva = va[F].astype(float), va["label_10"].to_numpy().astype(int)
    medians = Xtr.median()
    models, calib, val_scores, out = {}, {}, {}, {"best_params": best, "n_train": len(tr), "n_val": len(va),
                                                   "base_rate_train": float(ytr.mean()), "base_rate_val": float(yva.mean())}
    for kind in ["xgb", "rf", "logit"]:
        m, info = ml.fit_final(kind, best[kind], Xtr, ytr, Xva, yva)
        p_va = ml.predict(kind, m, Xva, medians)
        iso = fit_isotonic(p_va, yva)
        models[kind], calib[kind] = m, iso
        val_scores[kind] = p_va
        out[f"{kind}_val"] = {"auc": float(roc_auc_score(yva, p_va)), "pr_auc": float(average_precision_score(yva, p_va)),
                              "brier_raw": brier(p_va, yva), "brier_calibrated_in_sample": brier(iso.predict(p_va), yva),
                              **info}
        log_result("P5", "ml_final", kind, "validation", len(yva), "auc", out[f"{kind}_val"]["auc"],
                   out[f"{kind}_val"], seed_value=seed(1))
    comp_va = comp.score(va)["score"].to_numpy()
    out["composite_val_auc"] = float(roc_auc_score(yva, comp_va))
    out["H4_provisional_val"] = delong_test(yva, val_scores["xgb"], comp_va)
    thr_ml = float(np.quantile(calib["xgb"].predict(val_scores["xgb"]), 0.9))
    thr_raw = float(np.quantile(val_scores["xgb"], 0.9))
    thr_comp = float(np.quantile(comp_va, 0.9))
    pm, nm = top_decile_precision(yva, val_scores["xgb"], thr_raw)
    pc, nc = top_decile_precision(yva, comp_va, thr_comp)
    out["top_decile_val"] = {"ml_precision": pm, "ml_n": nm, "composite_precision": pc, "composite_n": nc,
                             "base_rate": float(yva.mean())}
    out["thresholds"] = {"ml_top_decile_raw": thr_raw, "ml_top_decile_calibrated": thr_ml, "composite_top_decile": thr_comp}
    out["reliability_val_xgb"] = reliability(calib["xgb"].predict(val_scores["xgb"]), yva).to_dict("records")
    # secondary target: volatility expansion (XGBoost, same best params)
    tv2 = tr[tr["vol_exp_label"].notna()]
    va2 = va[va["vol_exp_label"].notna()]
    if len(tv2) > 100 and tv2["vol_exp_label"].nunique() == 2:
        mv, _ = ml.fit_final("xgb", best["xgb"], tv2[F].astype(float), tv2["vol_exp_label"].astype(int).to_numpy(),
                             va2[F].astype(float), va2["vol_exp_label"].astype(int).to_numpy())
        pv = mv.predict_proba(va2[F].astype(float))[:, 1]
        out["vol_expansion_val_auc"] = float(roc_auc_score(va2["vol_exp_label"].astype(int), pv))
        models["xgb_volexp"] = mv
    # SHAP on a seeded validation sample
    n_s = min(int(c["models"]["shap"]["sample"]), len(va))
    sample = Xva.sample(n_s, random_state=seed(500))
    imp, sv = global_importance(models["xgb"], sample)
    imp.to_csv(METRICS / "p5_shap_importance.csv", index=False)
    pairs, verdict = interaction_ranking(models["xgb"], sample)
    pairs.head(200).to_csv(METRICS / "p5_shap_interactions_top200.csv", index=False)
    out["shap_top10"] = imp.head(10).to_dict("records")
    out["shap_novelty_verdict"] = verdict
    out["shap_fii_vs_dii"] = {g: float(imp.loc[imp.feature.str.startswith(g + "_"), "mean_abs_shap"].sum())
                              for g in ["fii", "dii", "pro", "client"]}
    FIGURES.mkdir(parents=True, exist_ok=True)
    for color in ["fii_fut_stk_cot", "oi_cot", "bbw_pct252"]:
        dependence_plot(sv, sample, "deliv_z20", color, FIGURES / f"shap_dependence_deliv_z20_by_{color}.png")
    # freeze
    blob = {"models": models, "calibrators": calib, "medians": medians, "thresholds": out["thresholds"],
            "best_params": best, "features": F}
    raw = pickle.dumps(blob)
    (MODELS / "ml_frozen.pkl").write_bytes(raw)
    out["model_freeze_sha256"] = hashlib.sha256(raw).hexdigest()
    save_json({"sha256": out["model_freeze_sha256"]}, METRICS / "model_freeze.json")
    save_json(out, METRICS / "p5_ml.json")
    log(f"P5: val AUC xgb {out['xgb_val']['auc']:.4f} rf {out['rf_val']['auc']:.4f} logit {out['logit_val']['auc']:.4f} "
        f"composite {out['composite_val_auc']:.4f}; novelty {verdict}")


if __name__ == "__main__":
    main()
