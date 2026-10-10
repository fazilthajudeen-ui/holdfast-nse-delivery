"""IMPI v2 panel: every Tier-1 (F&O) stock-day with v1 + v2 features, market features and labels.

    python -m impi.v2.panel

Labels (all from the v1 builder's base table or computed here; never used as features):
  label_10   triple-barrier success (v1 primary endpoint)
  label_xs10 1 if the stock's 10-day forward return beats that day's Tier-1 median (market-neutral)
  vol_exp    volatility expansion RV10(after)/RV10(before) >= 1.5
Writes data/processed/features/panel_t1.parquet (float32 features).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from impi.common import PROCESSED, config, log
from impi.features.build import FO_COLS, OUT
from impi.features.compression import compression_features
from impi.features.delivery import delivery_features
from impi.features.positioning import stock_positioning
from impi.features.vcp import EMPTY as VCP_EMPTY, VCPDetector
from impi.features.volume import volume_features
from impi.labels.vol_expansion import vol_expansion
from impi.transforms import pit_store as pit
from impi.v2.features_v2 import extra_features

PANEL = OUT / "panel_t1.parquet"
DROP = {"high20_prev", "atr14", "bbw", "in_fo", "vcp_pivot", "deliv_z20_prev"}


def build(end_date=None, out_path=None) -> pd.DataFrame:
    t0 = time.time()
    fcfg = config()["features"]
    fo = pd.read_parquet(pit.PIT / "fo_stock_adj.parquet", columns=["date", "symbol"] + FO_COLS)
    fo["symbol"] = fo["symbol"].astype(str)
    fo_by = {s: g.drop(columns="symbol") for s, g in fo.groupby("symbol")}
    syms = sorted(fo_by)
    del fo
    cols = ["date", "symbol", "series", "open", "high", "low", "close", "raw_close", "volume", "deliv_pct", "turnover_rs"]
    eq = pit.read("equity_adj", columns=cols, symbols=syms)
    eq["symbol"] = eq["symbol"].astype(str)
    eq = eq.sort_values(["symbol", "date"])
    frames = []
    for i, (sym, df) in enumerate(eq.groupby("symbol", sort=True), 1):
        df = df.reset_index(drop=True)
        for c_ in ["volume", "deliv_pct", "turnover_rs"]:
            df[c_] = df[c_].astype("float64")
        if len(df) < 260:
            continue
        df = df.merge(fo_by[sym], on="date", how="left").rename(columns={"call_oi_adj": "call_oi", "put_oi_adj": "put_oi"})
        f = pd.concat([delivery_features(df, fcfg), volume_features(df, fcfg), compression_features(df, fcfg),
                       stock_positioning(df, fcfg), extra_features(df)], axis=1)
        f = f.loc[:, ~f.columns.duplicated()]
        keep = (df["fut_oi_adj"].notna() & (df["series"] == "EQ")).to_numpy()
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            continue
        det = VCPDetector(df["high"], df["low"], df["close"], df["volume"], fcfg["vcp"])
        vcp = [det.pattern_at(int(t)) for t in idx]
        vl, _ = vol_expansion(df["close"].to_numpy(), idx, 10, 10, 1.5)
        part = f.iloc[idx].copy()
        part.insert(0, "date", df["date"].to_numpy()[idx])
        part.insert(1, "symbol", sym)
        for k in [k_ for k_ in VCP_EMPTY if k_ != "vcp_pivot"]:
            part[k] = [p[k] for p in vcp]
        part["vol_exp"] = vl
        part["raw_close"] = df["raw_close"].to_numpy()[idx]
        part["close"] = df["close"].to_numpy()[idx]
        frames.append(part.drop(columns=[c for c in DROP if c in part.columns]))
        if i % 100 == 0:
            log(f"  panel: {i}/{len(syms)} symbols")
    panel = pd.concat(frames, ignore_index=True)
    del frames, eq
    market = pd.read_parquet(OUT / "market.parquet")
    mcols = [c for c in market.columns if not c.endswith(("_share", "_share_104w", "_cot_104w", "_cot_d20")) and c != "available_at"]
    panel = panel.merge(market[mcols], on="date", how="left")
    base = pd.read_parquet(OUT / "base.parquet", columns=["date", "symbol", "eligible", "label_10", "ret_tb_10",
                                                          "fwd_10", "fwd_ex_10", "label_end", "liq_bucket",
                                                          "entry_date", "entry_px", "exit_date_10", "exit_px_10"])
    panel = panel.merge(base, on=["date", "symbol"], how="left")
    med = panel[panel["eligible"].fillna(False)].groupby("date")["fwd_10"].median()
    panel["label_xs10"] = np.where(panel["fwd_10"].notna(), (panel["fwd_10"] > panel["date"].map(med)).astype(float), np.nan)
    if end_date is not None:
        panel = panel[panel["date"] <= pd.Timestamp(end_date)]
    for c_ in panel.columns:
        if panel[c_].dtype == "float64" and c_ not in ("raw_close", "close", "entry_px", "exit_px_10"):
            panel[c_] = panel[c_].astype("float32")
    panel["eligible"] = panel["eligible"].fillna(False).astype(bool)
    panel.to_parquet(out_path or PANEL, index=False)
    log(f"panel: {panel.shape} in {time.time() - t0:.0f}s")
    return panel


def main():
    build()


if __name__ == "__main__":
    main()
