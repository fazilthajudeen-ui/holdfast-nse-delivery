"""Market regime features (one row per date): Nifty 50 trend (50d vs 200d SMA), India VIX percentile,
20-day FII flow proxy, advance/decline breadth.

FII flow (D-003): NSE publishes no historical daily FII/DII cash-market archive (cat_turnover does
not separate FII/DII and stops before 2026), so the 20-day FII flow is proxied by the 20-day change in
FII net-long share of stock + index futures OI.
"""
from __future__ import annotations

import pandas as pd


def regime_features(index: pd.DataFrame, breadth: pd.DataFrame, mkt: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["regime"]
    px = index.pivot_table(index="date", columns="index", values="close").sort_index()
    out = pd.DataFrame(index=px.index)
    n50 = px["nifty50"]
    sma_f = n50.rolling(c["sma_fast"], min_periods=c["sma_fast"]).mean()
    sma_s = n50.rolling(c["sma_slow"], min_periods=c["sma_slow"]).mean()
    out["nifty_above_200"] = (n50 > sma_s).astype(float).where(sma_s.notna())
    out["nifty_trend_50_200"] = sma_f / sma_s - 1
    out["vix_pct252"] = px["india_vix"].rolling(c["vix_pct_window"], min_periods=c["vix_pct_window"] // 2).rank(pct=True) * 100
    out["nifty500_ret20"] = px["nifty500"] / px["nifty500"].shift(20) - 1
    b = breadth.set_index("date").sort_index()
    ad = (b["advances"] - b["declines"]) / b["n"].where(b["n"] > 0)
    out["breadth10"] = ad.rolling(c["breadth_window"], min_periods=c["breadth_window"]).mean().reindex(out.index)
    m = mkt.set_index("date").sort_index()
    if {"fii_fut_stk_share", "fii_fut_idx_share"} <= set(m.columns):
        fii = m["fii_fut_stk_share"] + m["fii_fut_idx_share"]
        dii = m["dii_fut_stk_share"] + m["dii_fut_idx_share"]
        out["fii_flow20"] = (fii - fii.shift(c["flow_window"])).reindex(out.index)
        out["dii_flow20"] = (dii - dii.shift(c["flow_window"])).reindex(out.index)
    out.index.name = "date"
    return out.reset_index()
