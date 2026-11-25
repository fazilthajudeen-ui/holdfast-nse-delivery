"""Smoothed delivery regimes ("buyer zones") for the chart.

smoothed_rel(t) = trailing EMA of rel_deliv (delivery % minus the stock's 60-day normal), so a
value on day t uses only days up to t. Hysteresis turns it into zones:

    enter  Accumulation  when smoothed_rel >= +enter_pts,  leave when it falls below +exit_pts
    enter  Low-delivery  when smoothed_rel <= -enter_pts,  leave when it rises above -exit_pts

Descriptive only: it highlights stretches where buyers kept taking delivery above normal.
It is not part of the backtested result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACCUM, NONE, LOWDEL = 1, 0, -1


def smooth_rel(rel: pd.Series, span: int) -> pd.Series:
    return rel.ewm(span=span, adjust=False, min_periods=max(3, span // 2)).mean()


def regimes(smoothed: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """State per day (1 accumulation, -1 low delivery, 0 none) with hysteresis; causal."""
    out = np.zeros(len(smoothed), dtype=int)
    state = NONE
    for i, v in enumerate(smoothed):
        if np.isnan(v):
            state = NONE
        elif state == ACCUM:
            if v < exit_:
                state = NONE
        elif state == LOWDEL:
            if v > -exit_:
                state = NONE
        if state == NONE and not np.isnan(v):
            if v >= enter:
                state = ACCUM
            elif v <= -enter:
                state = LOWDEL
        out[i] = state
    return out


def zones(dates: pd.Series, state: np.ndarray) -> list[dict]:
    """Contiguous runs of non-zero state -> [{start, end, kind, days}]."""
    runs, start = [], None
    d = list(pd.to_datetime(dates))
    for i, s in enumerate(state):
        if start is None and s != NONE:
            start, kind = i, s
        elif start is not None and s != kind:
            runs.append((start, i - 1, kind))
            start, kind = (i, s) if s != NONE else (None, None)
    if start is not None:
        runs.append((start, len(state) - 1, kind))
    return [{"start": d[a].strftime("%Y-%m-%d"), "end": d[b].strftime("%Y-%m-%d"),
             "kind": "accumulation" if k == ACCUM else "low", "days": b - a + 1} for a, b, k in runs]
