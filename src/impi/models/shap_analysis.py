"""SHAP analysis of the XGBoost challenger (spec section 9): global importance, SHAP interaction
ranking of all feature pairs, and dependence plots for delivery x positioning.

Novelty check (pre-registered): the pairs (deliv_z20, fii_fut_stk_cot), (deliv_z20, oi_cot) and
(deliv_z20, bbw_pct252) - is each in the TOP QUARTILE of all feature pairs by mean |SHAP interaction|?
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

KEY_PAIRS = [("deliv_z20", "fii_fut_stk_cot"), ("deliv_z20", "oi_cot"), ("deliv_z20", "bbw_pct252")]


def global_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    import shap
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True), sv


def interaction_ranking(model, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import shap
    ex = shap.TreeExplainer(model)
    iv = ex.shap_interaction_values(X)                    # (n, f, f)
    m = np.abs(iv).mean(axis=0)
    cols = list(X.columns)
    rows = [{"f1": cols[i], "f2": cols[j], "mean_abs_interaction": float(m[i, j] + m[j, i])}
            for i, j in itertools.combinations(range(len(cols)), 2)]
    pairs = pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)
    pairs["rank"] = np.arange(1, len(pairs) + 1)
    pairs["percentile"] = 1 - (pairs["rank"] - 1) / len(pairs)
    q75 = pairs["mean_abs_interaction"].quantile(0.75)
    verdict = {}
    for a, b in KEY_PAIRS:
        hit = pairs[((pairs.f1 == a) & (pairs.f2 == b)) | ((pairs.f1 == b) & (pairs.f2 == a))]
        if hit.empty:
            verdict[f"{a}|{b}"] = {"present": False}
            continue
        r = hit.iloc[0]
        verdict[f"{a}|{b}"] = {"present": True, "rank": int(r["rank"]), "of": len(pairs),
                               "percentile": float(r["percentile"]),
                               "top_quartile": bool(r["mean_abs_interaction"] >= q75)}
    return pairs, verdict


def dependence_plot(sv: np.ndarray, X: pd.DataFrame, feature: str, color_by: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i = list(X.columns).index(feature)
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(X[feature], sv[:, i], c=X[color_by], cmap="coolwarm", s=6, alpha=0.6)
    fig.colorbar(sc, ax=ax, label=color_by)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value of {feature}")
    ax.set_title(f"SHAP dependence: {feature} coloured by {color_by}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
