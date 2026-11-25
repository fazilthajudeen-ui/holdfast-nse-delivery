"""Detect sharp price moves, score their delivery, measure what happened next.

Event (up):   close[t] / min(close[t-W .. t-1]) - 1 >= move_pct   (trough close -> close on t)
Event (down): close[t] / max(close[t-W .. t-1]) - 1 <= -move_pct
Cool-down:    after an event, the same stock cannot fire another same-direction event
              for `cooldown` trading days.
Scoring:      event_deliv     = sum(DELIV_QTY) / sum(TTL_TRD_QNTY) over trough+1 .. t
              event_rel_deliv = event_deliv - baseline_60 on the trough day (uses data < trough)
              tercile         = Low / Mid / High conviction by event_rel_deliv
Outcomes:     forward returns +5/+20/+60 days (raw and excess vs Nifty 500), round-trip
              (gave back >= 50% of the move within 60 days), max drawdown within 60 days.
              Horizons that run past the last date are NaN and flagged incomplete.

All trading-day arithmetic uses the market calendar (dates any stock traded).

    python -m src.events
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common import load_config, log, path

GROUP_LABELS = {3: ["Low", "Mid", "High"], 5: ["Q1 Low", "Q2", "Q3", "Q4", "Q5 High"]}


# ---------------------------------------------------------------- market data (wide)
@dataclass
class Market:
    dates: pd.DatetimeIndex
    symbols: list[str]
    close: np.ndarray        # dates x symbols, forward-filled inside each stock's life, NaN outside
    volume: np.ndarray       # 0 on non-trading days
    deliv: np.ndarray        # 0 on non-trading days / missing
    baseline: np.ndarray     # baseline_60, forward-filled
    bench_fwd: dict          # horizon -> np.ndarray (dates,) benchmark forward return
    bench_name: str
    analysis_start: pd.Timestamp


def load_market(features: pd.DataFrame | None = None, cfg: dict | None = None) -> Market:
    cfg = cfg or load_config()
    df = features if features is not None else pd.read_parquet(path("features"))
    wide = lambda col: df.pivot(index="date", columns="symbol", values=col).sort_index()
    close = wide("close")
    last_valid = close.apply(pd.Series.last_valid_index)
    close = close.ffill()
    close = close.where(close.index.values[:, None] <= last_valid.values[None, :])
    volume = wide("volume").fillna(0.0)
    deliv = wide("deliv_qty")
    deliv = deliv.where(volume > 0).fillna(0.0)
    # rows with volume but missing delivery: exclude from the delivery ratio entirely
    volume_for_deliv = wide("volume").where(wide("deliv_qty").notna()).fillna(0.0)
    baseline = wide("baseline_60").ffill()
    dates = close.index

    horizons = cfg["events"]["horizons"]
    bench_fwd, bench_name = {}, "median stock in universe"
    bench = None
    bp = path("benchmark")
    if bp.exists():
        b = pd.read_parquet(bp)
        if len(b):
            s = b.set_index("date")["nifty500"].reindex(dates).ffill()
            if s.notna().mean() > 0.95:
                bench, bench_name = s.values, "Nifty 500 index"
    for h in horizons:
        if bench is not None:
            fwd = np.full(len(dates), np.nan)
            fwd[:-h] = bench[h:] / bench[:-h] - 1
        else:
            c = close.values
            fwd_all = np.full_like(c, np.nan)
            fwd_all[:-h] = c[h:] / c[:-h] - 1
            with np.errstate(all="ignore"):
                fwd = np.nanmedian(fwd_all, axis=1)
        bench_fwd[h] = fwd

    analysis_start = dates.max() - pd.DateOffset(months=cfg["analysis_months"])
    return Market(dates, list(close.columns), close.values, volume_for_deliv.values, deliv.values,
                  baseline.values, bench_fwd, bench_name, analysis_start)


# ---------------------------------------------------------------- detection
def detect_moves(close: np.ndarray, move_pct: float, window: int, cooldown: int,
                 direction: str = "up", first_t: int = 0) -> list[tuple[int, int]]:
    """Return [(event_idx, trough_idx)] for one price series. Uses only data up to each t."""
    events, next_allowed = [], max(first_t, window)
    n = len(close)
    for t in range(max(window, first_t), n):
        if t < next_allowed or np.isnan(close[t]):
            continue
        past = close[t - window:t]
        if np.all(np.isnan(past)):
            continue
        # most recent extreme (reverse search) = the day the move actually started
        if direction == "up":
            m = t - 1 - int(np.nanargmin(past[::-1]))
            hit = close[t] / close[m] - 1 >= move_pct
        else:
            m = t - 1 - int(np.nanargmax(past[::-1]))
            hit = close[t] / close[m] - 1 <= -move_pct
        if hit:
            events.append((t, m))
            next_allowed = t + cooldown + 1
    return events


def assign_groups(values: pd.Series, n_groups: int = 3) -> pd.Series:
    labels = GROUP_LABELS.get(n_groups, [f"G{i + 1}" for i in range(n_groups)])
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    ok = values.notna()
    if ok.sum() >= n_groups:
        out[ok] = pd.qcut(values[ok].rank(method="first"), n_groups, labels=labels).astype(str)
    return out


def build_events(mkt: Market, move_pct: float, window: int, cooldown: int,
                 directions=("up", "down"), cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ecfg = cfg["events"]
    horizons, rt_days, giveback = ecfg["horizons"], ecfg["roundtrip_days"], ecfg["roundtrip_giveback"]
    first_t = int(np.searchsorted(mkt.dates.values, np.datetime64(mkt.analysis_start)))
    n_dates = len(mkt.dates)
    rows = []
    for j, sym in enumerate(mkt.symbols):
        c = mkt.close[:, j]
        last_idx = int(np.max(np.where(~np.isnan(c))[0]))
        for direction in directions:
            for t, m in detect_moves(c, move_pct, window, cooldown, direction, first_t):
                vol = mkt.volume[m + 1:t + 1, j].sum()
                ev_deliv = mkt.deliv[m + 1:t + 1, j].sum() / vol * 100 if vol > 0 else np.nan
                base = mkt.baseline[m, j]
                r = {
                    "symbol": sym, "direction": direction,
                    "start_date": mkt.dates[m], "event_date": mkt.dates[t],
                    "move_days": t - m, "trough_close": c[m], "event_close": c[t],
                    "move_pct": (c[t] / c[m] - 1) * 100,
                    "event_deliv": ev_deliv, "baseline_start": base,
                    "event_rel_deliv": ev_deliv - base,
                }
                for h in horizons:
                    ok = t + h <= min(last_idx, n_dates - 1)
                    ret = c[t + h] / c[t] - 1 if ok else np.nan
                    r[f"ret_{h}"] = ret * 100
                    r[f"exc_{h}"] = (ret - mkt.bench_fwd[h][t]) * 100 if ok else np.nan
                    r[f"complete_{h}"] = bool(ok)
                path_ = c[t:min(t + rt_days, last_idx) + 1]           # includes event day
                full = t + rt_days <= last_idx
                if direction == "up":
                    level = c[t] - giveback * (c[t] - c[m])
                    hit = bool(np.any(path_[1:] <= level))
                else:
                    level = c[t] + giveback * (c[m] - c[t])
                    hit = bool(np.any(path_[1:] >= level))
                r["round_trip"] = True if hit else (False if full else np.nan)
                r["max_dd_60"] = (np.min(path_ / np.maximum.accumulate(path_)) - 1) * 100
                r["status"] = "complete" if full else "incomplete"
                rows.append(r)
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev["round_trip"] = ev["round_trip"].astype("float")          # 1.0 / 0.0 / NaN
    ev = ev.dropna(subset=["event_rel_deliv"]).reset_index(drop=True)
    ev["group"] = pd.NA
    for d in ev["direction"].unique():
        mask = ev["direction"] == d
        ev.loc[mask, "group"] = assign_groups(ev.loc[mask, "event_rel_deliv"], ecfg["n_groups"])
    return ev.sort_values(["event_date", "symbol"]).reset_index(drop=True)


def main():
    cfg = load_config()
    e = cfg["events"]
    mkt = load_market(cfg=cfg)
    log(f"market: {len(mkt.dates)} days x {len(mkt.symbols)} stocks; analysis window from "
        f"{mkt.analysis_start:%Y-%m-%d}; benchmark = {mkt.bench_name}")
    ev = build_events(mkt, e["move_pct"], e["window"], e["cooldown"], cfg=cfg)
    ev.to_parquet(path("events"), index=False)

    print("EVENTS CHECK")
    for d, x in ev.groupby("direction"):
        print(f"  {d:>4}: {len(x)} events, {x['symbol'].nunique()} stocks, "
              f"complete(60d) {(x['status'] == 'complete').sum()}, "
              f"groups {x['group'].value_counts().to_dict()}")
    cuts = ev[ev.direction == "up"].groupby("group")["event_rel_deliv"].agg(["min", "max"]).round(1)
    print("  up-move tercile ranges of event_rel_deliv (pp):\n" + cuts.to_string())

    print("\nSPOT-CHECK: 10 random up-move events (verify against a chart on nseindia.com)")
    cols = ["symbol", "start_date", "event_date", "trough_close", "event_close", "move_pct",
            "event_deliv", "baseline_start", "group", "ret_60", "round_trip"]
    sample = ev[ev.direction == "up"].sample(min(10, (ev.direction == "up").sum()), random_state=7)
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(sample[cols].to_string(index=False))
    print("  (prices are split/bonus-adjusted to today's terms; see data/logs/corporate_actions.csv)")
    log(f"saved {path('events').name}")


if __name__ == "__main__":
    main()


def similar_pairs(ev: pd.DataFrame, tolerance_pp: float = 3.0, horizon: int = 60) -> pd.DataFrame:
    """Match completed High-tercile up-moves to completed Low-tercile up-moves of similar size
    (|move_pct difference| <= tolerance). Greedy one-to-one matching; no hand-picking."""
    done = ev[(ev["direction"] == "up") & ev[f"complete_{horizon}"]]
    hi = done[done["group"] == "High"].sort_values("move_pct")
    lo = done[done["group"] == "Low"].copy()
    pairs = []
    for _, h in hi.iterrows():
        if lo.empty:
            break
        gap = (lo["move_pct"] - h["move_pct"]).abs()
        j = gap.idxmin()
        if gap[j] <= tolerance_pp:
            l = lo.loc[j]
            pairs.append({"high_idx": h.name, "low_idx": j, "move_gap": gap[j],
                          "high_ret": h[f"ret_{horizon}"], "low_ret": l[f"ret_{horizon}"]})
            lo = lo.drop(index=j)
    return pd.DataFrame(pairs)
