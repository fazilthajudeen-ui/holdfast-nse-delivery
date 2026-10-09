"""Chronological splits with purging, and purged K-fold CV with embargo (G3, G4).

Purging: an event belongs to a split only if its whole label window [T, label_end] lies inside the
split, so no training label overlaps validation/test periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.common import config, d as as_date


def split_bounds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    s = config()["splits"]
    return {k: (pd.Timestamp(as_date(s[k][0])), pd.Timestamp(as_date(s[k][1])))
            for k in ("train", "validation", "test")}


def assign_split(dates: pd.Series, label_end: pd.Series | None = None) -> pd.Series:
    """'train' | 'validation' | 'test' | None (outside, or label window crosses the split end)."""
    out = pd.Series(None, index=dates.index, dtype=object)
    for name, (a, b) in split_bounds().items():
        m = (dates >= a) & (dates <= b)
        if label_end is not None:
            le = pd.to_datetime(label_end)
            m &= le.notna() & (le <= b)
        out[m] = name
    return out


def purged_kfold(dates: pd.Series, label_end: pd.Series, n_folds: int, embargo_days: int,
                 calendar: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous time-ordered folds. For each test fold [a, b]: training events are dropped when
    their label window overlaps [a, b] (purge) or when they start within `embargo_days` trading
    days after b (embargo)."""
    d = pd.to_datetime(dates).to_numpy()
    le = pd.to_datetime(label_end).to_numpy()
    order = np.argsort(d, kind="stable")
    folds = np.array_split(order, n_folds)
    cal = np.asarray(calendar, dtype="datetime64[ns]")
    out = []
    for f in folds:
        a, b = d[f].min(), d[f].max()
        bi = np.searchsorted(cal, b)
        emb_end = cal[min(bi + embargo_days, len(cal) - 1)]
        test_mask = np.zeros(len(d), bool)
        test_mask[f] = True
        overlap = (d <= b) & (le >= a)                  # label window intersects test window
        embargo = (d > b) & (d <= emb_end)
        train_mask = ~test_mask & ~overlap & ~embargo
        out.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask)))
    return out
