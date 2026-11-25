"""IMPI v2 extra features (post-hoc model development, AFTER the one-time v1 test).

Pure functions of ONE stock's history up to day T (no look-ahead; tested by truncation). They add
what v1 lacked: momentum/reversal at several horizons, distance from 52-week extremes, realised
volatility and its term structure, short-window delivery, turnover shocks and intraday location.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def extra_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype("float64")
    h, l, o = df["high"].astype("float64"), df["low"].astype("float64"), df["open"].astype("float64")
    v = df["volume"].astype("float64")
    tv = df["turnover_rs"].astype("float64")
    d = df["deliv_pct"].astype("float64")
    out = pd.DataFrame(index=df.index)
    for n in (1, 5, 60, 120):
        out[f"ret_{n}"] = c / c.shift(n) - 1
    out["dist_52w_high"] = c / h.rolling(252, min_periods=126).max() - 1
    out["dist_52w_low"] = c / l.rolling(252, min_periods=126).min() - 1
    lr = np.log(c / c.shift(1))
    rv5 = lr.rolling(5, min_periods=5).std()
    rv20 = lr.rolling(20, min_periods=15).std()
    rv60 = lr.rolling(60, min_periods=40).std()
    out["rv20"] = rv20 * np.sqrt(252)
    out["rv_5_20"] = rv5 / rv20
    out["rv_20_60"] = rv20 / rv60
    out["rv20_pct252"] = rv20.rolling(252, min_periods=126).rank(pct=True) * 100
    d5 = d.rolling(5, min_periods=4).mean()
    d20 = d.rolling(20, min_periods=15).mean()
    base = d5.shift(1)
    out["deliv_z5"] = (d5 - base.rolling(252, min_periods=126).mean()) / base.rolling(252, min_periods=126).std()
    out["deliv_trend_5_20"] = d5 - d20
    out["deliv_pct_rank252"] = d20.rolling(252, min_periods=126).rank(pct=True) * 100
    ltv = np.log(tv.where(tv > 0))
    out["turnover_z20"] = (ltv - ltv.shift(1).rolling(60, min_periods=40).mean()) / ltv.shift(1).rolling(60, min_periods=40).std()
    rng = (h - l).where(h > l)
    out["clv5"] = ((c - l) / rng).rolling(5, min_periods=4).mean()          # where the close sits in the day's range
    out["gap_1"] = o / c.shift(1) - 1
    out["up_days_20"] = (c.diff() > 0).astype(float).rolling(20, min_periods=20).mean()
    return out
