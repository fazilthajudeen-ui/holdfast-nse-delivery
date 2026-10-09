"""Benjamini-Hochberg FDR across the PRE-REGISTERED family only (spec section 7)."""
from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvals, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Returns (adjusted q-values, reject mask). NaN p-values are kept as NaN and never rejected."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    qv = np.full(len(p), np.nan)
    rej = np.zeros(len(p), bool)
    m = ok.sum()
    if m == 0:
        return qv, rej
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty(m)
    out[order] = adj
    qv[ok] = out
    rej[ok] = out <= q
    return qv, rej
