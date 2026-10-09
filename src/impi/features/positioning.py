"""Positioning features (spec section 5, adapted per D-001/D-002/D-004).

Stock level (Tier 1, from the F&O bhavcopy; OI in SHARES, split-adjusted - lot-size changes do not
move it): futures OI change, a COT-style 0-100 index of the OI level, its acceleration and
persistence, price-OI build-up quadrant, divergence magnitude, annualised futures basis, put-call OI.

Market level (participant-wise OI, contracts): for each participant x segment, net long as a SHARE of
total segment OI (invariant to lot-size revisions, D-004), then a 52-week COT index, 5-day change,
change-of-change and persistence. These are the same for every stock on a given day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cot_index(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or window // 2
    lo = x.rolling(window, min_periods=mp).min()
    hi = x.rolling(window, min_periods=mp).max()
    rng = hi - lo
    return (100 * (x - lo) / rng.where(rng > 0)).clip(0, 100)


def run_length(x: pd.Series) -> pd.Series:
    """Signed count of consecutive same-direction daily changes (+n up-run, -n down-run, 0 flat)."""
    s = np.sign(x.diff()).fillna(0).to_numpy()
    out = np.zeros(len(s))
    for i in range(1, len(s)):
        if s[i] == 0:
            out[i] = 0
        elif s[i] == np.sign(out[i - 1]) and out[i - 1] != 0:
            out[i] = out[i - 1] + s[i]
        else:
            out[i] = s[i]
    out[np.isnan(x.to_numpy())] = np.nan
    return pd.Series(out, index=x.index)


def stock_positioning(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """df: one stock's equity rows (date-sorted) with F&O columns merged (NaN when not in F&O):
    fut_oi_adj, call_oi, put_oi, near_settle, near_expiry, next_settle, next_expiry, raw_close, close."""
    c = cfg["positioning"]
    w, a, dw = c["cot_window"], c["accel_window"], c["divergence_window"]
    oi = df["fut_oi_adj"].astype("float64").where(lambda s: s > 0)
    close = df["close"].astype("float64")
    out = pd.DataFrame(index=df.index)
    out["oi_chg5"] = oi / oi.shift(5) - 1
    out["oi_chg20"] = oi / oi.shift(dw) - 1
    cot = cot_index(oi, w)
    out["oi_cot"] = cot
    out["oi_cot_d5"] = cot - cot.shift(a)
    out["oi_cot_dd"] = out["oi_cot_d5"] - out["oi_cot_d5"].shift(a)
    out["oi_persist"] = run_length(oi)
    ret20 = close / close.shift(dw) - 1
    pos, neg = ret20 > 0, ret20 < 0
    oiu, oid = out["oi_chg20"] > 0, out["oi_chg20"] < 0
    valid = ret20.notna() & out["oi_chg20"].notna()
    out["long_buildup"] = (pos & oiu).astype(float).where(valid)
    out["short_covering"] = (pos & oid).astype(float).where(valid)
    out["short_buildup"] = (neg & oiu).astype(float).where(valid)
    out["long_unwinding"] = (neg & oid).astype(float).where(valid)
    dcot = cot - cot.shift(dw)
    disagree = np.sign(dcot) * np.sign(ret20) < 0
    out["oi_div_mag"] = (disagree.astype(float) * dcot.abs()).where(dcot.notna() & ret20.notna())
    out["price_oi_align"] = (np.sign(ret20) * out["oi_chg20"]).where(valid)
    # annualised basis from same-day RAW prices (ratio-safe); next month when near month is too close
    spot = df["raw_close"].astype("float64")
    dte_n = (pd.to_datetime(df["near_expiry"]) - df["date"]).dt.days
    dte_x = (pd.to_datetime(df["next_expiry"]) - df["date"]).dt.days
    use_next = dte_n < c["basis_min_dte"]
    fut = df["near_settle"].where(~use_next, df["next_settle"]).astype("float64")
    dte = dte_n.where(~use_next, dte_x).astype("float64")
    out["basis_ann"] = ((fut / spot - 1) * 365 / dte.where(dte > 0)).clip(-1, 1)
    out["log_pcr_oi"] = np.log((df["put_oi"].astype("float64") + 1) / (df["call_oi"].astype("float64") + 1)).where(
        df["call_oi"].notna() | df["put_oi"].notna())
    out["in_fo"] = df["fut_oi_adj"].notna().astype(float)
    return out


def market_positioning(po: pd.DataFrame, cfg: dict, long_window: bool = False) -> pd.DataFrame:
    """po: participant_oi table (date, participant, fut_idx_long, ...). Returns one row per date."""
    c = cfg["positioning"]
    w = c["cot_window_long"] if long_window else c["cot_window"]
    a = c["accel_window"]
    tag = "_104w" if long_window else ""
    wide = {}
    total = po[po["participant"] == "TOTAL"].set_index("date").sort_index()
    for p in c["participants"]:
        sub = po[po["participant"] == p].set_index("date").sort_index()
        for s in c["segments"]:
            denom = total[f"{s}_long"].reindex(sub.index)
            share = (sub[f"{s}_long"] - sub[f"{s}_short"]) / denom.where(denom > 0)
            cot = cot_index(share, w)
            key = f"{p.lower()}_{s}"
            wide[f"{key}_share{tag}"] = share
            wide[f"{key}_cot{tag}"] = cot
            if not long_window:
                wide[f"{key}_cot_d5"] = cot - cot.shift(a)
                wide[f"{key}_cot_dd"] = wide[f"{key}_cot_d5"] - wide[f"{key}_cot_d5"].shift(a)
                wide[f"{key}_persist"] = run_length(cot)
                wide[f"{key}_cot_d20"] = cot - cot.shift(c.get("divergence_window", 20))
    out = pd.DataFrame(wide)
    out.index.name = "date"
    return out.reset_index()
