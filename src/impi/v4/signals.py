"""IMPI v4 signals (pre-registered in pre_registered_v4.md). Rules only - nothing is trained.

    python -m impi.v4.signals          # writes features/signals_v4.parquet and features/market_v4.parquet

Every signal for day T uses only data published by the T 23:59 decision point (G1); outcome columns are
computed from later bars and are NEVER used as signals.

Stock level (eligible Tier-1 F&O stock-days, from the v3 panel):
  D1  deliv_z20 (delivery vs its own history)            outcome move20 = |fwd_20| / (rv20 * sqrt(20/252))
  D2  sds = mean over the last 10 days of (delivery % - previous-60-day median) x sign(daily return)
      (+ = high delivery on up days = accumulation, - = on down days = distribution)   outcome fwd_20
Event level:
  H2' all 20-day-high breakouts (B1, Tier 1, eligible); exposure ret_20 > 0 and oi_chg20 < 0
      outcome label_20 (triple barrier +2/-1 ATR, 20 days)
  V1  loose VCP breakout, fixed rule k=5, max depth 0.5, volume x1.25, base 20-60 days (Tier 1, eligible);
      exposure deliv_z20_prev > 0; outcome label_10
Market level:
  C1  FII stock-futures COT index <= 10 (bottom decile of its 52-week range)
      outcome n500_fwd60 = Nifty 500 close(T+61) / close(T+1) - 1
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from impi.common import config, log
from impi.features.build import OUT
from impi.features.delivery import delivery_features
from impi.features.vcp import VCPDetector
from impi.labels.events import apply_cooldown
from impi.transforms import pit_store as pit
from impi.v3.panel import PANEL_V3

SIGNALS = OUT / "signals_v4.parquet"
MARKET_V4 = OUT / "market_v4.parquet"
V1_RULE = {"fractal_k": 5, "max_depth": 0.5, "volume_mult": 1.25, "base_min_days": 20, "base_max_days": 60}
SDS_WINDOW, SDS_BASE = 10, 60
C1_THRESHOLD = 10.0


def signed_delivery(df: pd.DataFrame) -> pd.Series:
    """D2 score for ONE stock (rows sorted by date): mean over the last 10 days of the delivery gap
    (delivery % minus its previous-60-day median) signed by that day's price change."""
    d = df["deliv_pct"].astype("float64")
    base = d.shift(1).rolling(SDS_BASE, min_periods=40).median()
    sign = np.sign(df["close"].astype("float64").diff())
    return ((d - base) * sign).rolling(SDS_WINDOW, min_periods=7).mean()


def loose_vcp_triggers(df: pd.DataFrame) -> np.ndarray:
    """V1 trigger bar indices for ONE stock with the fixed loose rule (same cooldown as STRICT)."""
    fcfg = config()["features"]
    cfg = {**fcfg["vcp"], **V1_RULE}
    v = df["volume"].astype("float64")
    vr = (v / v.shift(1).rolling(20, min_periods=15).mean()).to_numpy()
    det = VCPDetector(df["high"], df["low"], df["close"], df["volume"], cfg)
    trig = np.array([t for t, _ in det.triggers(vr)], int)
    cooldown = config()["labels"]["events"]["strict"]["cooldown_days"]
    trig = apply_cooldown(trig, cooldown)
    return trig[(df["series"].to_numpy()[trig] == "EQ")] if len(trig) else trig


def build(out_dir=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    fcfg = config()["features"]
    p = pd.read_parquet(PANEL_V3, columns=["date", "symbol", "eligible", "deliv_z20", "rv20", "ret_20", "oi_chg20",
                                           "fwd_20", "label_end_20", "raw_close"])
    p["symbol"] = p["symbol"].astype(str)
    syms = sorted(p["symbol"].unique())
    eq = pit.read("equity_adj", columns=["date", "symbol", "series", "high", "low", "close", "volume", "deliv_pct"],
                  symbols=syms)
    eq["symbol"] = eq["symbol"].astype(str)
    sds_parts, v1_parts = [], []
    for i, (sym, g) in enumerate(eq.sort_values(["symbol", "date"]).groupby("symbol", sort=True), 1):
        g = g.reset_index(drop=True)
        for c in ["deliv_pct", "volume"]:
            g[c] = g[c].astype("float64")
        sds_parts.append(pd.DataFrame({"date": g["date"], "symbol": sym, "sds": signed_delivery(g)}))
        trig = loose_vcp_triggers(g)
        if len(trig):
            dz = delivery_features(g, fcfg)["deliv_z20_prev"].to_numpy()
            v1_parts.append(pd.DataFrame({"date": g["date"].to_numpy()[trig], "symbol": sym, "v1_event": True,
                                          "v1_exposed": dz[trig] > 0, "v1_dz_prev": dz[trig]}))
        if i % 100 == 0:
            log(f"  v4 signals: {i}/{len(syms)} symbols")
    sds = pd.concat(sds_parts, ignore_index=True)
    p = p.merge(sds, on=["date", "symbol"], how="left")
    p["move20"] = p["fwd_20"].abs() / (p["rv20"] * np.sqrt(20 / 252))
    p.loc[~np.isfinite(p["move20"]), "move20"] = np.nan
    # B1 breakouts (H2') and their 20-day labels from the v1 event table
    ev = pd.read_parquet(OUT / "events.parquet", columns=["date", "symbol", "family", "tier1", "eligible", "label_10",
                                                           "label_20"])
    b1 = ev[(ev["family"] == "all") & ev["tier1"]][["date", "symbol", "label_20"]].copy()
    b1["symbol"] = b1["symbol"].astype(str)
    b1["b1_event"] = True
    p = p.merge(b1, on=["date", "symbol"], how="left")
    p["b1_event"] = p["b1_event"].astype("boolean").fillna(False).astype(bool)
    p["h2_exposed"] = p["b1_event"] & (p["ret_20"] > 0) & (p["oi_chg20"] < 0)
    # V1 loose-VCP events with the 10-day triple-barrier label of the base table
    v1 = pd.concat(v1_parts, ignore_index=True) if v1_parts else pd.DataFrame(columns=["date", "symbol", "v1_event", "v1_exposed", "v1_dz_prev"])
    base = pd.read_parquet(OUT / "base.parquet", columns=["date", "symbol", "label_10"])
    base["symbol"] = base["symbol"].astype(str)
    p = p.merge(v1, on=["date", "symbol"], how="left").merge(base, on=["date", "symbol"], how="left")
    p["v1_event"] = p["v1_event"].astype("boolean").fillna(False).astype(bool)
    p["v1_exposed"] = p["v1_exposed"].astype("boolean").fillna(False).astype(bool)
    for c in ["deliv_z20", "sds", "move20", "fwd_20", "label_20", "label_10"]:
        p[c] = p[c].astype("float64")
    p.to_parquet(SIGNALS if out_dir is None else out_dir / SIGNALS.name, index=False)
    # market level (C1)
    mk = pd.read_parquet(OUT / "market.parquet", columns=["date", "fii_fut_stk_cot"])
    idx = pit.read("index")
    n5 = idx[idx["index"] == "nifty500"].set_index("date")["close"].sort_index()
    fwd = (n5.shift(-61) / n5.shift(-1) - 1).rename("n500_fwd60")
    mk = mk.merge(fwd.reset_index(), on="date", how="left")
    mk["c1_on"] = mk["fii_fut_stk_cot"] <= C1_THRESHOLD
    mk.to_parquet(MARKET_V4 if out_dir is None else out_dir / MARKET_V4.name, index=False)
    log(f"signals_v4: {p.shape}, B1 {int(p['b1_event'].sum()):,}, V1 {int(p['v1_event'].sum()):,}, "
        f"market {mk.shape} in {time.time() - t0:.0f}s")
    return p, mk


def main():
    build()


if __name__ == "__main__":
    main()
