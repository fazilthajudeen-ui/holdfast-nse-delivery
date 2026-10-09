"""Event triggers, cooldown and event-level exclusions (spec sections 2 and 6).

All three event families are evaluated on the close of day T with entry at the T+1 open:
  B1 all_breakouts : close_T > highest high of T-20..T-1
  BROAD            : all_breakout AND BBW percentile < 30 on >= 5 of the 10 days T-10..T-1
  STRICT           : VCP detector trigger (features/vcp.py)
Cooldown (D-015) is applied to the RAW triggers of each family: a trigger within `cooldown` trading
rows of the previous kept trigger of the same stock and family is dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def breakout_mask(close: pd.Series, high20_prev: pd.Series) -> np.ndarray:
    return (close > high20_prev).fillna(False).to_numpy()


def compression_regime(bbw_pct: pd.Series, pct_max: float, min_days: int, lookback: int) -> np.ndarray:
    comp = (bbw_pct < pct_max).astype(float).where(bbw_pct.notna())
    cnt = comp.shift(1).rolling(lookback, min_periods=lookback).sum()
    return (cnt >= min_days).fillna(False).to_numpy()


def apply_cooldown(idx: np.ndarray, cooldown: int) -> np.ndarray:
    keep, last = [], -10 ** 9
    for i in np.sort(np.asarray(idx, int)):
        if i - last > cooldown:
            keep.append(i)
            last = i
    return np.array(keep, int)


def earnings_mask(row_gidx: np.ndarray, row_dates: np.ndarray, res_gidx: np.ndarray, res_known: np.ndarray,
                  window: int, decision_offset: pd.Timedelta) -> np.ndarray:
    """True where day T is within +/- window trading days of a results date, using only information
    known at the T decision point for FUTURE results (G1): a future meeting counts only if its
    intimation timestamp <= T + decision offset."""
    out = np.zeros(len(row_gidx), bool)
    decision = pd.to_datetime(row_dates) + decision_offset
    for e, known in zip(res_gidx, res_known):
        after = (row_gidx >= e) & (row_gidx <= e + window)          # results already out
        before = (row_gidx >= e - window) & (row_gidx < e)            # results upcoming
        if before.any():
            if pd.isna(known):
                known_ok = np.zeros(len(row_gidx), bool)
            else:
                known_ok = np.asarray(decision >= pd.Timestamp(known))
            before = before & known_ok
        out |= after | before
    return out
