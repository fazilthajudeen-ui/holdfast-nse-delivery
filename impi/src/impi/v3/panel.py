"""IMPI v3 panel = v2 panel + option-implied features + 20-day targets.

    python -m impi.v3.panel

Reads data/processed/features/panel_t1.parquet (v2) and pit/options_iv.parquet; writes panel_v3.parquet.
Option features are functions of each stock's own option history up to day T (no look-ahead):
  iv30, iv_rv (iv30 / rv20), iv_rv_spread, iv_term (next/near ATM IV), iv_skew (put wing - call wing),
  iv_smile (mean wing - iv30), iv_chg5, iv_chg20, iv_pct252, iv_z60, dte_near, log_pcr_vol,
  log_opt_fut_vol, opt_vol_z60.
New targets (never features):
  fwd_20      close(T+20) / open(T+1) - 1            (same construction as fwd_10)
  label_xs20  1 if fwd_20 beats that day's eligible Tier-1 median
  label_20    triple barrier +2/-1 ATR, 20 days       (v1 builder, secondary horizon)
  label_end_20 date of bar T+20 (purging for the 20-day targets)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from impi.common import log
from impi.features.build import OUT
from impi.transforms import pit_store as pit
from impi.v2.panel import PANEL

PANEL_V3 = OUT / "panel_v3.parquet"
OPT = ["iv30", "iv_rv", "iv_rv_spread", "iv_term", "iv_skew", "iv_smile", "iv_chg5", "iv_chg20", "iv_pct252",
       "iv_z60", "dte_near", "log_pcr_vol", "log_opt_fut_vol", "opt_vol_z60"]


def option_features(iv: pd.DataFrame) -> pd.DataFrame:
    """Per-stock time-series features on the stock's own option-day series (sorted by date)."""
    iv = iv.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = iv.groupby("symbol", sort=False)
    out = iv[["date", "symbol"]].copy()
    s = iv["iv30"]
    out["iv30"] = s
    out["iv_term"] = iv["iv_next"] / iv["iv_near"]
    out["iv_skew"] = iv["iv_put_wing"] - iv["iv_call_wing"]
    out["iv_smile"] = (iv["iv_put_wing"] + iv["iv_call_wing"]) / 2 - s
    out["iv_chg5"] = s - g["iv30"].shift(5)
    out["iv_chg20"] = s - g["iv30"].shift(20)
    out["iv_pct252"] = g["iv30"].transform(lambda x: x.rolling(252, min_periods=126).rank(pct=True) * 100)
    prev = g["iv30"].shift(1)
    m60 = prev.groupby(iv["symbol"]).transform(lambda x: x.rolling(60, min_periods=40).mean())
    s60 = prev.groupby(iv["symbol"]).transform(lambda x: x.rolling(60, min_periods=40).std())
    out["iv_z60"] = (s - m60) / s60
    out["dte_near"] = iv["dte_near"]
    calls, puts, futs = iv["call_contracts"].astype(float), iv["put_contracts"].astype(float), iv["fut_contracts"].astype(float)
    out["log_pcr_vol"] = np.log((puts + 1) / (calls + 1))
    out["log_opt_fut_vol"] = np.log((calls + puts + 1) / (futs + 1))
    lv = np.log(calls + puts + 1)
    lprev = lv.groupby(iv["symbol"]).shift(1)
    lm = lprev.groupby(iv["symbol"]).transform(lambda x: x.rolling(60, min_periods=40).mean())
    ls = lprev.groupby(iv["symbol"]).transform(lambda x: x.rolling(60, min_periods=40).std())
    out["opt_vol_z60"] = (lv - lm) / ls
    return out


def forward_20(symbols: list[str]) -> pd.DataFrame:
    """fwd_20 and label_end_20 per stock-day from the adjusted equity series (bars T+1 open .. T+20 close)."""
    eq = pit.read("equity_adj", columns=["date", "symbol", "open", "close"], symbols=symbols)
    eq["symbol"] = eq["symbol"].astype(str)
    eq = eq.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = eq.groupby("symbol", sort=False)
    entry = g["open"].shift(-1)
    exit_px = g["close"].shift(-20)
    eq["fwd_20"] = (exit_px / entry - 1).astype("float64")
    eq["label_end_20"] = g["date"].shift(-20)
    return eq[["date", "symbol", "fwd_20", "label_end_20"]]


def build(out_path=None) -> pd.DataFrame:
    t0 = time.time()
    p = pd.read_parquet(PANEL)
    iv = pd.read_parquet(pit.PIT / "options_iv.parquet")
    iv["symbol"] = iv["symbol"].astype(str)
    of = option_features(iv)
    p = p.merge(of, on=["date", "symbol"], how="left")
    p["iv_rv"] = p["iv30"] / p["rv20"]
    p["iv_rv_spread"] = p["iv30"] - p["rv20"]
    base = pd.read_parquet(OUT / "base.parquet", columns=["date", "symbol", "label_20"])
    p = p.merge(base, on=["date", "symbol"], how="left")
    p = p.merge(forward_20(sorted(p["symbol"].unique())), on=["date", "symbol"], how="left")
    med = p[p["eligible"]].groupby("date")["fwd_20"].median()
    p["label_xs20"] = np.where(p["fwd_20"].notna(), (p["fwd_20"] > p["date"].map(med)).astype(float), np.nan)
    for c in OPT + ["fwd_20", "label_20", "label_xs20"]:
        # ratios/z-scores with a zero denominator (e.g. rv20 = 0 for a halted stock) are undefined -> missing
        p[c] = p[c].replace([np.inf, -np.inf], np.nan).astype("float32")
    p.to_parquet(out_path or PANEL_V3, index=False)
    cov = {c: round(float(p.loc[p["eligible"], c].notna().mean()), 3) for c in OPT}
    log(f"panel_v3: {p.shape} in {time.time() - t0:.0f}s; option-feature coverage (eligible rows): {cov}")
    return p


def main():
    build()


if __name__ == "__main__":
    main()
