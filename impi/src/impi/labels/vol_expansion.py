"""Volatility-expansion label (secondary target): RV10(T+10) / RV10(T) >= 1.5.
RV10(T) = std of daily log returns over bars T-9..T; RV10(T+10) = over bars T+1..T+10."""
from __future__ import annotations

import numpy as np


def vol_expansion(close, t_idx, window: int = 10, horizon: int = 10, threshold: float = 1.5):
    c = np.asarray(close, float)
    r = np.full(len(c), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(c[1:] / c[:-1])
    t = np.asarray(t_idx, int)
    n = len(c)
    ratio = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        if ti - window + 1 < 1 or ti + horizon >= n:
            continue
        past = r[ti - window + 1:ti + 1]
        fut = r[ti + 1:ti + horizon + 1]
        if np.isfinite(past).sum() < window - 1 or np.isfinite(fut).sum() < horizon - 1:
            continue
        sp = np.nanstd(past, ddof=1)
        if sp > 0:
            ratio[i] = np.nanstd(fut, ddof=1) / sp
    label = np.where(np.isfinite(ratio), (ratio >= threshold).astype(float), np.nan)
    return label, ratio
