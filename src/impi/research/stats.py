"""Shared statistical tools: two-proportion z-test, month-block bootstrap, DeLong AUC test,
power / minimum-detectable-effect calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Pooled two-sided z-test of p1 - p2."""
    if min(n1, n2) == 0:
        return {"p1": np.nan, "p2": np.nan, "diff": np.nan, "z": np.nan, "p": np.nan, "n1": n1, "n2": n2}
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    pv = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return {"p1": p1, "p2": p2, "diff": p1 - p2, "z": z, "p": pv, "n1": n1, "n2": n2}


def month_block_bootstrap(df: pd.DataFrame, stat_fn, n: int, rng: np.random.Generator,
                          date_col: str = "date") -> dict:
    """Cluster bootstrap that resamples whole calendar months (keeps cross-sectional and short-run
    serial dependence between overlapping events). Returns estimate, CI, two-sided p vs 0."""
    est = stat_fn(df)
    months = pd.to_datetime(df[date_col]).dt.to_period("M")
    groups = {m: idx for m, idx in df.groupby(months).indices.items()}
    keys = list(groups)
    if len(keys) < 2 or not np.isfinite(est):
        return {"estimate": est, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n_boot": 0}
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        v = stat_fn(df.iloc[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    if len(vals) < 10:
        return {"estimate": est, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n_boot": len(vals)}
    p = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return {"estimate": est, "ci_low": float(np.quantile(vals, 0.025)), "ci_high": float(np.quantile(vals, 0.975)),
            "p": float(min(1.0, p)), "n_boot": len(vals)}


# ------------------------------------------------------------------ DeLong (fast, Sun & Xu 2014)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1)
        i = k
    out = np.empty(n)
    out[j] = t + 1
    return out


def _delong_cov(preds: np.ndarray, m: int):
    """preds: (k, n) with the first m columns positives."""
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Paired DeLong test of AUC(a) - AUC(b). Two-sided p."""
    y = np.asarray(y).astype(int)
    ok = np.isfinite(score_a) & np.isfinite(score_b)
    y, a, b = y[ok], np.asarray(score_a)[ok], np.asarray(score_b)[ok]
    order = np.argsort(-y, kind="stable")          # positives first
    m = int(y.sum())
    if m == 0 or m == len(y):
        return {"auc_a": np.nan, "auc_b": np.nan, "diff": np.nan, "se": np.nan, "z": np.nan, "p": np.nan}
    preds = np.vstack([a[order], b[order]])
    aucs, cov = _delong_cov(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(var, 0))
    z = (aucs[0] - aucs[1]) / se if se > 0 else np.nan
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(aucs[0] - aucs[1]),
            "se": float(se), "z": float(z), "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "n": int(len(y)), "n_pos": m}


def auc_se_hanley(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    return float(np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)))


def mde_two_proportions(p: float, n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    if min(n1, n2) == 0:
        return np.nan
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)))


def mde_auc_paired(auc: float, n_pos: int, n_neg: int, rho: float = 0.5, alpha: float = 0.05,
                   power: float = 0.8) -> float:
    if min(n_pos, n_neg) < 2:
        return np.nan
    se = auc_se_hanley(auc, n_pos, n_neg)
    se_diff = np.sqrt(2 * se ** 2 * (1 - rho))
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se_diff)
