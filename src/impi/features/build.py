"""Build the feature store, events and labels in one memory-light pass over stocks (P2).

    python -m impi.features.build

Outputs (data/processed/features/):
  market.parquet  one row per date: participant COT features (52w + 104w), divergence, regime
  events.parquet  every event (family all/broad/strict, tier 1/2) with stock + market features,
                  exclusion flags, triple-barrier labels (10/20/60d), vol-expansion label, forward returns
  base.parquet    every Tier-1 stock-day (eligibility flags + 10/20/60d labels) for the B0 base rate,
                  liquidity buckets and the backtest benchmark
Every row carries available_at = date + latest release time of its inputs (<= decision point, G1).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from impi.common import PROCESSED, config, log, save_json
from impi.features.compression import compression_features
from impi.features.delivery import delivery_features
from impi.features.divergence import market_divergence
from impi.features.positioning import market_positioning, stock_positioning
from impi.features.regime import regime_features
from impi.features.vcp import EMPTY as VCP_EMPTY, VCPDetector
from impi.features.volume import volume_features
from impi.labels.events import apply_cooldown, breakout_mask, compression_regime, earnings_mask
from impi.labels.triple_barrier import triple_barrier
from impi.labels.vol_expansion import vol_expansion
from impi.transforms import pit_store as pit

OUT = PROCESSED / "features"
FO_COLS = ["fut_oi_adj", "call_oi_adj", "put_oi_adj", "near_settle", "near_expiry", "next_settle", "next_expiry"]
USED = ["equity_bhav", "delivery_mto", "fo_bhav", "participant_oi", "index_close"]


def build_market(fcfg: dict) -> pd.DataFrame:
    po = pit.read("participant_oi")
    mkt = market_positioning(po, fcfg)
    mkt104 = market_positioning(po, fcfg, long_window=True)
    index = pit.read("index")
    breadth = pit.read("breadth")
    reg = regime_features(index, breadth, mkt, fcfg)
    mdiv = market_divergence(mkt, reg.set_index("date")["nifty500_ret20"], fcfg)
    m = reg.merge(mkt, on="date", how="left").merge(mkt104, on="date", how="left").merge(mdiv, on="date", how="left")
    return m.sort_values("date").reset_index(drop=True)


def relevant_symbols(fo_syms: set, min_turnover: float, window: int) -> set:
    """F&O underlyings plus every symbol whose rolling median turnover ever exceeds the Tier-2 floor
    (computed year by year to save memory; the exact point-in-time test is applied per row later)."""
    keep = set()
    for p in sorted((pit.PIT / "equity_raw").glob("year=*/part.parquet")):
        t = pd.read_parquet(p, columns=["date", "symbol", "series", "turnover_rs"])
        t = t[t["series"] == "EQ"]
        t["symbol"] = t["symbol"].astype(str)
        t = t.sort_values(["symbol", "date"])
        med = t.groupby("symbol")["turnover_rs"].rolling(window, min_periods=40).median()
        mx = med.groupby(level=0).max()
        keep |= set(mx[mx > min_turnover].index)
        keep |= fo_syms & set(t["symbol"].unique())
    return keep


def load_equity(syms: set, cols: list[str]) -> pd.DataFrame:
    cat = pd.CategoricalDtype(sorted(syms))
    parts = []
    for p in sorted((pit.PIT / "equity_adj").glob("year=*/part.parquet")):
        e = pd.read_parquet(p, columns=cols)
        e["symbol"] = e["symbol"].astype(str)
        e = e[e["symbol"].isin(syms)]
        e["symbol"] = e["symbol"].astype(cat)
        for c in ["volume", "deliv_pct", "turnover_rs"]:
            e[c] = e[c].astype("float32")
        parts.append(e)
    eq = pd.concat(parts, ignore_index=True)
    return eq.sort_values(["symbol", "date"]).reset_index(drop=True)


def main():
    t0 = time.time()
    cfg = config()
    fcfg, lcfg, ucfg = cfg["features"], cfg["labels"], cfg["universe"]
    for ds in USED:
        pit.assert_usable_at_t(ds)
    avail_offset = max(pit.release_offset(ds) for ds in USED)
    dec_off = pit.decision_offset()
    OUT.mkdir(parents=True, exist_ok=True)

    market = build_market(fcfg)
    market["available_at"] = market["date"] + avail_offset
    market.to_parquet(OUT / "market.parquet", index=False)
    log(f"market features: {market.shape}")

    cal = np.array(sorted(pit.read("breadth", columns=["date"])["date"].unique()), dtype="datetime64[ns]")
    data_end = cal[-1]
    idx = pit.read("index")
    n500 = idx[idx["index"] == "nifty500"].set_index("date").sort_index()
    n500_open = n500["open"].where(n500["open"] > 0, n500["close"])
    n500_close = n500["close"]
    ban = pit.read("ban")
    ban_set = set(zip(ban["symbol"].astype(str), ban["date"].values.astype("datetime64[ns]")))
    results = pd.read_parquet(pit.REFERENCE / "results_dates.parquet")
    results = results.assign(gidx=np.searchsorted(cal, results["meeting_date"].values.astype("datetime64[ns]")))
    res_by_sym = {s: g for s, g in results.groupby("symbol")}
    sectors = pd.read_parquet(pit.REFERENCE / "sector_map.parquet").set_index("symbol")["sector"].to_dict()

    fo = pd.read_parquet(pit.PIT / "fo_stock_adj.parquet",
                         columns=["date", "symbol"] + FO_COLS)
    fo["symbol"] = fo["symbol"].astype(str)
    fo_by_sym = {s: g.drop(columns="symbol") for s, g in fo.groupby("symbol")}
    del fo
    min_t2 = float(ucfg["tier2"]["min_median_turnover_rs"])
    syms = relevant_symbols(set(fo_by_sym), min_t2, ucfg["tier2"]["turnover_window"])
    log(f"symbols to process: {len(syms)} (F&O underlyings ever: {len(fo_by_sym)})")

    cols = ["date", "symbol", "series", "open", "high", "low", "close", "raw_close", "volume",
            "deliv_pct", "turnover_rs"]
    eq = load_equity(syms, cols)
    first_seen = eq.groupby("symbol", observed=True)["date"].min()
    last_seen = eq.groupby("symbol", observed=True)["date"].max()
    global_start = pd.Timestamp(cal[0])
    log(f"equity rows loaded: {len(eq):,}")

    vcfg = fcfg["vcp"]
    tb = lcfg["triple_barrier"]
    ev_frames, base_frames = [], []
    stats = {"symbols": 0, "strict_raw": 0, "broad_raw": 0, "all_raw": 0}
    for n_done, (sym, df) in enumerate(eq.groupby("symbol", sort=True, observed=True), 1):
        df = df.reset_index(drop=True)
        df["symbol"] = df["symbol"].astype(str)
        for c_ in ["volume", "deliv_pct", "turnover_rs"]:
            df[c_] = df[c_].astype("float64")
        n = len(df)
        if n < fcfg["min_history_days"] + 2:
            continue
        stats["symbols"] += 1
        f_fo = fo_by_sym.get(sym)
        if f_fo is not None:
            df = df.merge(f_fo, on="date", how="left")
        else:
            for c in FO_COLS:
                df[c] = np.nan
        df = df.rename(columns={"call_oi_adj": "call_oi", "put_oi_adj": "put_oi"})
        feats = pd.concat([delivery_features(df, fcfg), volume_features(df, fcfg),
                           compression_features(df, fcfg), stock_positioning(df, fcfg)], axis=1)
        dates = df["date"].to_numpy(dtype="datetime64[ns]")
        gidx = np.searchsorted(cal, dates)
        # ---- eligibility (event-level exclusions)
        is_eq = (df["series"] == "EQ").to_numpy()
        hist_ok = np.arange(n) >= fcfg["min_history_days"]
        price_ok = (df["raw_close"] >= ucfg["exclusions"]["min_raw_price"]).to_numpy()
        fs = first_seen[sym]
        listing_ok = np.ones(n, bool) if fs == global_start else \
            (gidx - np.searchsorted(cal, np.datetime64(fs, "ns"))) >= ucfg["exclusions"]["min_listing_days"]
        nxt = np.minimum(gidx + 1, len(cal) - 1)
        banned = np.array([((sym, d) in ban_set) or ((sym, cal[j]) in ban_set) for d, j in zip(dates, nxt)])
        r = res_by_sym.get(sym)
        earn = earnings_mask(gidx, dates, r["gidx"].to_numpy(), r["known_at"].to_numpy(),
                             ucfg["exclusions"]["earnings_window_days"], dec_off) if r is not None else np.zeros(n, bool)
        tier1 = df["fut_oi_adj"].notna().to_numpy()
        tv60 = df["turnover_rs"].rolling(ucfg["tier2"]["turnover_window"], min_periods=40).median().to_numpy()
        tier2 = is_eq & (tv60 > min_t2)
        eligible = is_eq & hist_ok & price_ok & listing_ok & ~banned & ~earn
        # ---- triggers
        close = df["close"]
        all_raw = np.flatnonzero(breakout_mask(close, feats["high20_prev"]) & is_eq)
        comp = compression_regime(feats["bbw_pct252"], lcfg["events"]["broad"]["bbw_pct_max"],
                                  lcfg["events"]["broad"]["min_days"], lcfg["events"]["broad"]["lookback_days"])
        broad_raw = np.flatnonzero(breakout_mask(close, feats["high20_prev"]) & comp & is_eq)
        det = VCPDetector(df["high"], df["low"], df["close"], df["volume"], vcfg)
        strict_trig = det.triggers(feats["vol_ratio20"].to_numpy())
        strict_raw = np.array([t for t, _ in strict_trig if is_eq[t]], int)
        stats["all_raw"] += len(all_raw)
        stats["broad_raw"] += len(broad_raw)
        stats["strict_raw"] += len(strict_raw)
        fam = {"all": apply_cooldown(all_raw, lcfg["events"]["all_breakouts"]["cooldown_days"]),
               "broad": apply_cooldown(broad_raw, lcfg["events"]["broad"]["cooldown_days"]),
               "strict": apply_cooldown(strict_raw, lcfg["events"]["strict"]["cooldown_days"])}
        stops = pd.Timestamp(last_seen[sym]) < pd.Timestamp(data_end)
        o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
        atr = feats["atr14"].to_numpy()

        def labels_for(t_idx):
            out = {}
            for hz in tb["horizons"]:
                res = triple_barrier(o, h, l, c, atr, t_idx, hz, tb["upper_atr"], tb["lower_atr"], stops)
                out[f"label_{hz}"] = res["label"]
                out[f"ret_tb_{hz}"] = res["ret"]
                out[f"exitk_{hz}"] = res["exit_k"]
                out[f"mfe_{hz}"] = res["mfe"]
                out[f"mae_{hz}"] = res["mae"]
            entry = np.where(t_idx + 1 < n, o[np.minimum(t_idx + 1, n - 1)], np.nan)
            entry_date = np.where(t_idx + 1 < n, dates[np.minimum(t_idx + 1, n - 1)], np.datetime64("NaT"))
            out["entry_date"] = entry_date
            out["entry_px"] = entry
            ph = tb["primary_horizon"]
            ek = out[f"exitk_{ph}"]
            xi = t_idx + np.nan_to_num(ek, nan=0).astype(int)
            out[f"exit_date_{ph}"] = np.where(np.isfinite(ek) & (xi < n), dates[np.minimum(xi, n - 1)], np.datetime64("NaT"))
            out[f"exit_px_{ph}"] = entry * (1 + out[f"ret_tb_{ph}"])
            for hz in lcfg["forward_returns"]:
                j = t_idx + hz
                okj = j < n
                px = np.where(okj, c[np.minimum(j, n - 1)], c[n - 1] if stops else np.nan)
                xd = np.where(okj, dates[np.minimum(j, n - 1)], dates[n - 1] if stops else np.datetime64("NaT"))
                fr = px / entry - 1
                io = n500_open.reindex(pd.to_datetime(entry_date)).to_numpy()
                ic = n500_close.reindex(pd.to_datetime(xd)).to_numpy()
                out[f"fwd_{hz}"] = fr
                out[f"fwd_ex_{hz}"] = fr - (ic / io - 1)
            # label end date (for purging) = date of the exit bar of the primary horizon (or T+h)
            k = out[f"exitk_{tb['primary_horizon']}"]
            kk = np.where(np.isfinite(k), t_idx + np.nan_to_num(k, nan=0).astype(int), t_idx + tb["primary_horizon"])
            out["label_end"] = np.where(kk < n, dates[np.minimum(kk, n - 1)], np.datetime64("NaT"))
            kk60 = t_idx + max(tb["horizons"])
            out["label_end_60"] = np.where(kk60 < n, dates[np.minimum(kk60, n - 1)], np.datetime64("NaT"))
            return out

        # ---- base table (Tier 1 rows)
        t1 = np.flatnonzero(tier1 & is_eq)
        if len(t1):
            lab = labels_for(t1)
            base_frames.append(pd.DataFrame({
                "date": dates[t1], "symbol": sym, "eligible": eligible[t1],
                "log_turnover60": feats["log_turnover60"].to_numpy()[t1],
                "label_10": lab["label_10"], "ret_tb_10": lab["ret_tb_10"], "exitk_10": lab["exitk_10"],
                "label_20": lab["label_20"], "label_60": lab["label_60"],
                "fwd_10": lab["fwd_10"], "fwd_ex_10": lab["fwd_ex_10"], "entry_date": lab["entry_date"],
                "entry_px": lab["entry_px"], "exit_date_10": lab["exit_date_10"], "exit_px_10": lab["exit_px_10"],
                "label_end": lab["label_end"]}).astype({"eligible": bool}))
        # ---- events
        for family, tidx in fam.items():
            if len(tidx) == 0:
                continue
            keep_tier1 = tier1[tidx]
            keep_tier2 = tier2[tidx] & (family != "all")
            sel = tidx[keep_tier1 | keep_tier2]
            if len(sel) == 0:
                continue
            ev = feats.iloc[sel].copy()
            ev.insert(0, "date", dates[sel])
            ev.insert(1, "symbol", sym)
            ev.insert(2, "family", family)
            ev["tier1"] = tier1[sel]
            ev["tier2"] = tier2[sel]
            ev["eligible"] = eligible[sel]
            ev["excl_ban"] = banned[sel]
            ev["excl_earnings"] = earn[sel]
            ev["excl_price"] = ~price_ok[sel]
            ev["excl_listing"] = ~np.asarray(listing_ok)[sel]
            ev["excl_history"] = ~hist_ok[sel]
            ev["raw_close"] = df["raw_close"].to_numpy()[sel]
            vcp = [det.pattern_at(int(t)) for t in sel]
            for k_ in VCP_EMPTY:
                ev[k_] = [v[k_] for v in vcp]
            lab = labels_for(sel)
            for k_, v in lab.items():
                ev[k_] = v
            vl, vr = vol_expansion(c, sel, lcfg["vol_expansion"]["rv_window"], lcfg["vol_expansion"]["horizon"],
                                   lcfg["vol_expansion"]["threshold"])
            ev["vol_exp_label"] = vl
            ev["vol_exp_ratio"] = vr
            ev["sector"] = sectors.get(sym, "Unknown")
            ev_frames.append(ev)
        if n_done % 200 == 0:
            log(f"  processed {n_done}/{len(syms)} symbols, {sum(len(x) for x in ev_frames):,} event rows")
    del eq
    events = pd.concat(ev_frames, ignore_index=True)
    events = events.merge(market, on="date", how="left", suffixes=("", "_mkt"))
    events["available_at"] = events["date"] + avail_offset
    assert (events["available_at"] <= events["date"] + dec_off).all(), "G1: feature released after decision point"
    base = pd.concat(base_frames, ignore_index=True)
    # point-in-time liquidity buckets: terciles of 60d median turnover among Tier-1 stocks each day
    pct = base.groupby("date")["log_turnover60"].rank(pct=True)
    base["liq_bucket"] = np.where(pct.isna(), None, np.where(pct <= 1 / 3, "small", np.where(pct <= 2 / 3, "mid", "large")))
    events = events.merge(base[["date", "symbol", "liq_bucket"]], on=["date", "symbol"], how="left")
    # Tier-2-only events have no Tier-1 bucket (NaN); costs treat them as 'small' (conservative)
    events.to_parquet(OUT / "events.parquet", index=False)
    base.to_parquet(OUT / "base.parquet", index=False)
    summary = {
        "symbols_processed": stats["symbols"], "raw_triggers": {k: v for k, v in stats.items() if k != "symbols"},
        "events_by_family_tier": {f"{a}|tier1={b}": int(v) for (a, b), v in
                                  events.groupby(["family", "tier1"]).size().items()},
        "events_eligible_tier1": events[events["tier1"] & events["eligible"]].groupby("family").size().to_dict(),
        "base_rows": len(base), "base_eligible": int(base["eligible"].sum()),
        "seconds": round(time.time() - t0, 1)}
    save_json(summary, OUT / "build_summary.json")
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
