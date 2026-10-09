"""Delivery features (spec section 5). Input: ONE stock's daily rows sorted by date.
All windows end at T (inclusive) and baselines end at T-1 -> no look-ahead (G1)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def delivery_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["delivery"]
    d = df["deliv_pct"].astype("float64")
    m = d.rolling(c["mean_window"], min_periods=int(c["mean_window"] * 0.75)).mean()
    base = m.shift(1)
    mu = base.rolling(c["z_window"], min_periods=c["z_min_periods"]).mean()
    sd = base.rolling(c["z_window"], min_periods=c["z_min_periods"]).std()
    z = (m - mu) / sd.where(sd > 1e-9)
    ret20 = df["close"] / df["close"].shift(20) - 1
    flag = ret20.abs() < c["accumulation_abs_ret"]
    out = pd.DataFrame(index=df.index)
    out["deliv_pct"] = d
    out["deliv_z20"] = z
    out["deliv_z20_prev"] = z.shift(1)              # formation window, excludes the trigger day (H1)
    out["deliv_accum_flag"] = flag.astype("float64").where(ret20.notna())
    out["deliv_accum"] = z.where(flag, 0.0).where(z.notna() & ret20.notna())
    out["ret_20"] = ret20
    return out
