"""ML challenger (spec section 9): XGBoost (primary), Random Forest, logistic regression - same
features, same events. Hyper-parameters are chosen by purged 5-fold CV (10-day embargo) on
train + validation; the final model is fit on TRAIN with early stopping on VALIDATION PR-AUC and then
calibrated (isotonic) on VALIDATION. The test set is never used here (G4). Every configuration tried
is appended to results/results_log.csv (G5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from impi.common import config, log_result, rng, seed

_P = ["fii", "dii", "pro", "client"]
_S = ["fut_idx", "fut_stk"]
FEATURES_V1 = (
    ["deliv_pct", "deliv_z20", "deliv_accum", "deliv_accum_flag",
     "vol_z", "updown_vol_log", "vol_ratio20", "log_amihud20", "log_turnover60",
     "bbw_pct252", "atr_ratio", "nr7_count10", "range_contraction",
     "vcp_valid", "vcp_n", "vcp_depth_first", "vcp_depth_last", "vcp_base_len", "vcp_vol_trend",
     "oi_chg5", "oi_chg20", "oi_cot", "oi_cot_d5", "oi_cot_dd", "oi_persist", "long_buildup",
     "short_covering", "short_buildup", "long_unwinding", "oi_div_mag", "price_oi_align", "basis_ann",
     "log_pcr_oi"]
    + [f"{p}_{s}_{k}" for p in _P for s in _S for k in ("cot", "cot_d5", "cot_dd", "persist", "div_mag")]
    + ["nifty_above_200", "nifty_trend_50_200", "vix_pct252", "breadth10", "fii_flow20", "dii_flow20", "ret_20"]
)


def class_weight(y: np.ndarray) -> float:
    pos = max(int(y.sum()), 1)
    neg = int(len(y) - y.sum())
    return float(min(neg / pos, config()["models"]["class_weight_cap"]))


def sample_xgb(r: np.random.Generator) -> dict:
    c = config()["models"]["xgboost"]
    u = lambda lo, hi: float(r.uniform(lo, hi))
    return {"max_depth": int(r.integers(c["max_depth"][0], c["max_depth"][1] + 1)),
            "learning_rate": float(np.exp(r.uniform(np.log(c["learning_rate"][0]), np.log(c["learning_rate"][1])))),
            "n_estimators": int(r.integers(100, c["n_estimators_max"] + 1)),
            "subsample": u(*c["subsample"]), "colsample_bytree": u(*c["colsample_bytree"]),
            "min_child_weight": u(*c["min_child_weight"]), "reg_alpha": u(*c["reg_alpha"]),
            "reg_lambda": u(*c["reg_lambda"])}


def sample_rf(r: np.random.Generator) -> dict:
    c = config()["models"]["random_forest"]
    return {"n_estimators": int(r.integers(c["n_estimators"][0], c["n_estimators"][1] + 1)),
            "max_depth": int(r.integers(c["max_depth"][0], c["max_depth"][1] + 1)),
            "max_features": float(r.uniform(*c["max_features"]))}


def make_model(kind: str, params: dict, spw: float, n_estimators: int | None = None):
    if kind == "xgb":
        p = dict(params)
        if n_estimators is not None:
            p["n_estimators"] = n_estimators
        return XGBClassifier(**p, objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
                             scale_pos_weight=spw, random_state=seed(1), n_jobs=4, verbosity=0)
    if kind == "rf":
        return RandomForestClassifier(**params, class_weight={0: 1.0, 1: spw}, min_samples_leaf=20,
                                      random_state=seed(2), n_jobs=3)
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=params["C"], class_weight={0: 1.0, 1: spw}, max_iter=2000))
    raise ValueError(kind)


def _prep(X: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "rf":
        return X.fillna(X.median()).fillna(0.0)       # RF cannot take NaN; medians of the fitting fold
    return X


def cv_score(kind: str, params: dict, X: pd.DataFrame, y: np.ndarray, folds) -> dict:
    pr, auc = [], []
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = make_model(kind, params, class_weight(y[tr]))
        Xtr = _prep(X.iloc[tr], kind)
        Xte = X.iloc[te]
        if kind == "rf":
            Xte = Xte.fillna(X.iloc[tr].median()).fillna(0.0)
        m.fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        pr.append(average_precision_score(y[te], p))
        auc.append(roc_auc_score(y[te], p))
    return {"cv_pr_auc": float(np.mean(pr)) if pr else np.nan, "cv_auc": float(np.mean(auc)) if auc else np.nan,
            "folds": len(pr)}


def search(kind: str, X: pd.DataFrame, y: np.ndarray, folds, phase: str = "P5") -> tuple[dict, pd.DataFrame]:
    c = config()["models"]
    r = rng(100 + {"xgb": 1, "rf": 2, "logit": 3}[kind])
    if kind == "xgb":
        grid = [sample_xgb(r) for _ in range(c["xgboost"]["n_iter"])]
    elif kind == "rf":
        grid = [sample_rf(r) for _ in range(c["random_forest"]["n_iter"])]
    else:
        grid = [{"C": C} for C in c["logistic"]["C"]]
    rows = []
    for i, params in enumerate(grid):
        s = cv_score(kind, params, X, y, folds)
        rows.append({"kind": kind, "config_id": i, **params, **s})
        log_result(phase, f"ml_search_{kind}", f"{kind}_{i}", "train+validation(cv)", len(y), "cv_pr_auc",
                   s["cv_pr_auc"], {**params, "cv_auc": s["cv_auc"]}, seed_value=seed(100))
    res = pd.DataFrame(rows)
    best = res.sort_values("cv_pr_auc", ascending=False).iloc[0]
    best_params = {k: best[k] for k in grid[0]}
    for k, v in best_params.items():
        if isinstance(grid[0][k], int):
            best_params[k] = int(v)
    return best_params, res


def fit_final(kind: str, params: dict, Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray):
    """Fit on train; XGBoost uses early stopping on validation PR-AUC (aucpr)."""
    spw = class_weight(ytr)
    if kind == "xgb":
        m = make_model("xgb", params, spw, n_estimators=config()["models"]["xgboost"]["n_estimators_max"])
        m.set_params(early_stopping_rounds=config()["models"]["xgboost"]["early_stopping_rounds"])
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m, {"best_iteration": int(m.best_iteration)}
    m = make_model(kind, params, spw)
    m.fit(_prep(Xtr, kind), ytr)
    return m, {}


def predict(kind: str, model, X: pd.DataFrame, train_medians: pd.Series | None = None) -> np.ndarray:
    if kind == "rf":
        X = X.fillna(train_medians).fillna(0.0)
    return model.predict_proba(X)[:, 1]
