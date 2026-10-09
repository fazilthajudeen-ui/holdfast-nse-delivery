"""Triple-barrier labels (Lopez de Prado), spec section 6.

Entry = open(T+1). Upper = entry + 2.0 x ATR14(T); lower = entry - 1.0 x ATR14(T); time barrier h days.
label = 1 if the upper barrier is touched before the lower one within h trading days (bars T+1..T+h).
  * both touched on the same bar -> 0 (D-016, conservative);
  * exit fills: barrier price, or the open when a later bar gaps through the barrier;
  * time expiry -> 0, exit at close(T+h);
  * not enough bars: if the stock stops trading before the data end (delisted/suspended) -> 0 with exit
    at its last close (D-017); if the data simply ends -> NaN (incomplete, excluded).
MFE/MAE are measured over the full horizon window (bars T+1..T+h) relative to entry.
"""
from __future__ import annotations

import numpy as np


def triple_barrier(o, h, l, c, atr, t_idx, horizon: int, up_mult: float, lo_mult: float,
                   stops_trading: bool) -> dict[str, np.ndarray]:
    o, h, l, c, atr = (np.asarray(x, float) for x in (o, h, l, c, atr))
    t = np.asarray(t_idx, int)
    n = len(c)
    m = len(t)
    label = np.full(m, np.nan)
    exit_k = np.full(m, np.nan)
    exit_px = np.full(m, np.nan)
    mfe = np.full(m, np.nan)
    mae = np.full(m, np.nan)
    has_entry = (t + 1 < n)
    entry = np.where(has_entry, o[np.minimum(t + 1, n - 1)], np.nan)
    a = atr[t]
    ok = has_entry & np.isfinite(entry) & (entry > 0) & np.isfinite(a) & (a > 0)
    up = entry + up_mult * a
    lo = entry - lo_mult * a
    resolved = ~ok
    hi_run = np.full(m, -np.inf)
    lo_run = np.full(m, np.inf)
    for k in range(1, horizon + 1):
        j = t + k
        inb = ok & (j < n)
        jj = np.minimum(j, n - 1)
        hj, lj, oj = h[jj], l[jj], o[jj]
        hi_run = np.where(inb & np.isfinite(hj), np.maximum(hi_run, hj), hi_run)
        lo_run = np.where(inb & np.isfinite(lj), np.minimum(lo_run, lj), lo_run)
        act = inb & ~resolved
        hit_lo = act & np.isfinite(lj) & (lj <= lo)
        hit_up = act & np.isfinite(hj) & (hj >= up) & ~hit_lo
        if k == 1:
            px_lo, px_up = lo, up
        else:
            px_lo = np.where(np.isfinite(oj) & (oj <= lo), oj, lo)
            px_up = np.where(np.isfinite(oj) & (oj >= up), oj, up)
        label = np.where(hit_lo, 0.0, np.where(hit_up, 1.0, label))
        exit_px = np.where(hit_lo, px_lo, np.where(hit_up, px_up, exit_px))
        exit_k = np.where(hit_lo | hit_up, k, exit_k)
        resolved = resolved | hit_lo | hit_up
    full = ok & (t + horizon < n)
    expire = full & ~resolved
    label = np.where(expire, 0.0, label)
    exit_px = np.where(expire, c[np.minimum(t + horizon, n - 1)], exit_px)
    exit_k = np.where(expire, horizon, exit_k)
    short = ok & ~full & ~resolved
    if stops_trading:
        label = np.where(short, 0.0, label)
        exit_px = np.where(short, c[n - 1], exit_px)
        exit_k = np.where(short, (n - 1) - t, exit_k)
    # MFE/MAE only when the whole window exists (or the stock stopped trading)
    win_ok = full | (ok & ~full & stops_trading)
    mfe = np.where(win_ok & np.isfinite(hi_run), hi_run / entry - 1, np.nan)
    mae = np.where(win_ok & np.isfinite(lo_run), lo_run / entry - 1, np.nan)
    ret = exit_px / entry - 1
    return {"label": label, "exit_k": exit_k, "ret": ret, "mfe": mfe, "mae": mae,
            "entry": entry, "upper": up, "lower": lo}
