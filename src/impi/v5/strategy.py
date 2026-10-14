"""IMPI v5 - breakout strategy improvement study (pre-registered in pre_registered_v5.md). Rules only.

A strategy variant = event set x market filter x stock filter x exit rule (fixed grid, 120 variants).
Every filter uses only data known by the T 23:59 decision point (G1); outcome columns are used only to
build the exit (exit date / exit price), exactly as in the v1 backtests.

  event sets   B1     every 20-day-high breakout (Tier 1, eligible)
               BROAD  B1 after compression (the v1 main event set)
  market       M0     none
               M1     Nifty 50 above its 200-day average (nifty_above_200 == 1)
               M2     Nifty 500 up over the last 20 days (nifty500_ret20 > 0)
  stock        S0     none
               S1     close within 10% of its 52-week high (dist_52w_high >= -0.10)
               S2     120-day return in the top 30% of that day's Tier-1 stocks (rs120 >= 0.70)
               S3     S1 and S2
  exit         TB10 / TB20 / TB60   triple barrier +2/-1 ATR within 10 / 20 / 60 trading days
               T20 / T60            time exit at the close of T+20 / T+60
Entry at the T+1 open, at most 20 equal-weight positions, full Indian costs (backtest/executor.py).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from impi.features.build import OUT
from impi.v3.panel import PANEL_V3
from impi.validation.splits import assign_split

EVENT_SETS = {"B1": "all", "BROAD": "broad"}
MARKET = ["M0", "M1", "M2"]
STOCK = ["S0", "S1", "S2", "S3"]
EXITS = {"TB10": ("tb", 10), "TB20": ("tb", 20), "TB60": ("tb", 60), "T20": ("time", 20), "T60": ("time", 60)}
NEAR_HIGH, RS_MIN = -0.10, 0.70
BASELINES = ["B1|M0|S0|TB10", "BROAD|M0|S0|TB10"]       # the v1 backtest rules, re-run in this frame


def variants() -> list[str]:
    return ["|".join(v) for v in itertools.product(EVENT_SETS, MARKET, STOCK, EXITS)]


def load_events() -> pd.DataFrame:
    """Tier-1, eligible B1 and BROAD events with filter inputs and a split assigned on the 60-day label
    window (so every exit rule uses the same events and no exit crosses a split boundary)."""
    cols = ["date", "symbol", "family", "tier1", "eligible", "liq_bucket", "nifty_above_200", "nifty500_ret20",
            "entry_date", "entry_px", "label_end_60"]
    for hz in (10, 20, 60):
        cols += [f"label_{hz}", f"ret_tb_{hz}", f"exitk_{hz}", f"fwd_{hz}"]
    ev = pd.read_parquet(OUT / "events.parquet", columns=cols)
    ev = ev[ev["tier1"] & ev["eligible"] & ev["family"].isin(EVENT_SETS.values())].copy()
    ev["symbol"] = ev["symbol"].astype(str)
    p = pd.read_parquet(PANEL_V3, columns=["date", "symbol", "eligible", "dist_52w_high", "ret_120"])
    p = p[p["eligible"]].drop(columns="eligible")
    p["symbol"] = p["symbol"].astype(str)
    p["rs120"] = p.groupby("date")["ret_120"].rank(pct=True)
    ev = ev.merge(p[["date", "symbol", "dist_52w_high", "rs120"]], on=["date", "symbol"], how="left")
    ev["split"] = assign_split(ev["date"], ev["label_end_60"])
    return ev.reset_index(drop=True)


def select(ev: pd.DataFrame, variant: str) -> pd.DataFrame:
    es, m, s, _ = variant.split("|")
    d = ev[ev["family"] == EVENT_SETS[es]]
    if m == "M1":
        d = d[d["nifty_above_200"] == 1]
    elif m == "M2":
        d = d[d["nifty500_ret20"] > 0]
    near = d["dist_52w_high"] >= NEAR_HIGH
    strong = d["rs120"] >= RS_MIN
    if s == "S1":
        d = d[near]
    elif s == "S2":
        d = d[strong]
    elif s == "S3":
        d = d[near & strong]
    return d


def signals(d: pd.DataFrame, variant: str, closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Backtest signals (entry/exit date and price) for the variant's exit rule. Exit dates are the
    stock's own k-th trading day after T (the same bars the labels were computed on)."""
    kind, hz = EXITS[variant.split("|")[3]]
    ret = d[f"ret_tb_{hz}"] if kind == "tb" else d[f"fwd_{hz}"]
    k = d[f"exitk_{hz}"] if kind == "tb" else pd.Series(float(hz), index=d.index)
    xd = []
    for sym, t, kk in zip(d["symbol"], d["date"], k):
        idx = closes[sym].index if sym in closes else None
        if idx is None or not np.isfinite(kk):
            xd.append(pd.NaT)
            continue
        j = idx.searchsorted(t) + int(kk)
        xd.append(idx[min(j, len(idx) - 1)])     # past the stock's last bar = it stopped trading (D-017)
    s = pd.DataFrame({"date": d["date"].to_numpy(), "symbol": d["symbol"].to_numpy(),
                      "family": d["family"].to_numpy(), "entry_date": d["entry_date"].to_numpy(),
                      "exit_date": pd.to_datetime(pd.Series(xd)).to_numpy(), "entry_px": d["entry_px"].to_numpy(),
                      "exit_px": (d["entry_px"] * (1 + ret)).to_numpy(), "liq_bucket": d["liq_bucket"].to_numpy()})
    s["priority"] = 0.0                           # no ranking: same-day ties broken by the seeded random draw
    return s


def success_rate(d: pd.DataFrame, variant: str) -> float:
    """Share of events whose exit made money before costs (TB exits: the +2 ATR barrier was hit)."""
    kind, hz = EXITS[variant.split("|")[3]]
    if kind == "tb":
        return float(d[f"label_{hz}"].mean()) if len(d) else np.nan
    return float((d[f"fwd_{hz}"] > 0).mean()) if len(d) else np.nan
