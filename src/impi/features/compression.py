"""Price-compression features: Bollinger bandwidth percentile, ATR ratio, NR7 count, range contraction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = (df[x].astype("float64") for x in ("high", "low", "close"))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1, skipna=True)
    tr = tr.where(h.notna() & l.notna())
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def compression_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["compression"]
    close = df["close"].astype("float64")
    h, l = df["high"].astype("float64"), df["low"].astype("float64")
    ma = close.rolling(c["bb_window"], min_periods=c["bb_window"]).mean()
    sd = close.rolling(c["bb_window"], min_periods=c["bb_window"]).std(ddof=0)
    bbw = (2 * c["bb_k"] * sd) / ma
    out = pd.DataFrame(index=df.index)
    out["bbw"] = bbw
    out["bbw_pct252"] = bbw.rolling(c["bbw_pct_window"], min_periods=int(c["bbw_pct_window"] * 0.5)).rank(pct=True) * 100
    atr_s, atr_l = wilder_atr(df, c["atr_short"]), wilder_atr(df, c["atr_long"])
    out["atr14"] = atr_s
    out["atr_ratio"] = atr_s / atr_l
    rng = h - l
    nr = (rng <= rng.rolling(c["nr_window"], min_periods=c["nr_window"]).min()) & rng.notna()
    out["nr7_count10"] = nr.astype("float64").rolling(c["nr_count_window"], min_periods=c["nr_count_window"]).sum()
    rs = h.rolling(c["rc_short"], min_periods=c["rc_short"]).max() - l.rolling(c["rc_short"], min_periods=c["rc_short"]).min()
    rl = h.rolling(c["rc_long"], min_periods=c["rc_long"]).max() - l.rolling(c["rc_long"], min_periods=c["rc_long"]).min()
    out["range_contraction"] = rs / rl.where(rl > 0)
    out["high20_prev"] = h.shift(1).rolling(20, min_periods=20).max()
    return out
