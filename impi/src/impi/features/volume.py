"""Volume and price-impact features. Volume is split/bonus ADJUSTED (G6); turnover is in rupees
(unaffected by splits). Baselines end at T-1."""
from __future__ import annotations

import numpy as np
import pandas as pd


def volume_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    v = df["volume"].astype("float64")
    c = cfg["volume"]
    logv = np.log(v.where(v > 0))
    base = logv.shift(1)
    mu = base.rolling(c["z_window"], min_periods=c["z_min_periods"]).mean()
    sd = base.rolling(c["z_window"], min_periods=c["z_min_periods"]).std()
    out = pd.DataFrame(index=df.index)
    out["vol_z"] = (logv - mu) / sd.where(sd > 1e-9)
    close = df["close"].astype("float64")
    chg = close.diff()
    w = c["updown_window"]
    up = v.where(chg > 0, 0.0).rolling(w, min_periods=w).sum()
    dn = v.where(chg < 0, 0.0).rolling(w, min_periods=w).sum()
    out["updown_vol_log"] = np.log((up + 1.0) / (dn + 1.0))
    avg20_prev = v.shift(1).rolling(cfg["vcp"]["volume_avg_window"], min_periods=15).mean()
    out["vol_ratio20"] = v / avg20_prev
    # Amihud illiquidity: mean(|r| / rupee volume) over 20 days, scaled and logged
    r = close.pct_change().abs()
    tv = df["turnover_rs"].astype("float64")
    ami = (r / tv.where(tv > 0)).rolling(cfg["price_impact"]["amihud_window"], min_periods=15).mean()
    out["log_amihud20"] = np.log(ami * 1e9 + 1e-12)
    out["log_turnover60"] = np.log(tv.rolling(60, min_periods=40).median().where(lambda x: x > 0))
    return out
