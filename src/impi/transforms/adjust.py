"""Split / bonus / consolidation / demerger adjustment of prices AND quantities (G6).

    python -m impi.transforms.adjust

Method (D-006):
 1. NSE corporate-action records give the expected price factor f (split 10->2: f = 0.2).
    A record is APPLIED only when the observed opening gap g = open_T / close_(T-1) on the ex-date
    (or the next 2 trading days) is within +/-20% of f ("confirmed"). Unconfirmed records are logged.
 2. Demerger / scheme records: applied with f = g when g < 0.97 (the price really dropped); else logged.
 3. Gap-only: an opening gap < 0.6 with no NSE record within 5 trading days, on a day following a normal
    trading day, is snapped to a standard split/bonus ratio within 3% (gap_snap); everything else
    (incl. unexplained upward jumps and gaps after suspensions) is logged and NOT adjusted.
    If > 5% of stocks gap on one day it is a missing session, not corporate actions.
 Prices before the adjustment date are multiplied by the product of later factors; quantities
 (volume, delivery qty, F&O OI in shares) are divided by it. Ratios (delivery %, basis, PCR) are
 unaffected. Raw (unadjusted) close is kept for the Rs 20 price filter (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.common import METRICS, log, save_json
from impi.transforms.pit_store import PIT, REFERENCE, read

TOL_CONFIRM = 0.20
GAP_LO, GAP_HI = 0.6, 1.6
SNAP_TOL = 0.03
STANDARD = [1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 1 / 20, 2 / 5, 2 / 3, 3 / 5, 2.0, 5.0, 10.0]
MASS_SHARE = 0.05
DEMERGER_MAX_GAP = 0.97      # with an NSE demerger record, an opening drop > 3% is the demerger (D-006)
PRICE_COLS = ["open", "high", "low", "close", "last", "prev_close"]
QTY_COLS = ["volume", "deliv_qty"]


def _year_files(name: str):
    return sorted((PIT / name).glob("year=*/part.parquet"))


def _symbol_dtype(name: str) -> pd.CategoricalDtype:
    syms = set()
    for p in _year_files(name):
        syms |= set(pd.read_parquet(p, columns=["symbol"])["symbol"].astype(str).unique())
    return pd.CategoricalDtype(sorted(syms))


def gap_table() -> pd.DataFrame:
    """Memory-light: one global categorical symbol dtype (no millions of Python strings)."""
    cat = _symbol_dtype("equity_raw")
    parts = []
    for p in _year_files("equity_raw"):
        e = pd.read_parquet(p, columns=["date", "symbol", "open", "close"])
        e["symbol"] = e["symbol"].astype(str).astype(cat)
        parts.append(e)
    g = pd.concat(parts, ignore_index=True)
    del parts
    g = g.sort_values(["symbol", "date"]).reset_index(drop=True)
    g["prev_close_own"] = g.groupby("symbol", observed=True)["close"].shift(1)
    o = g["open"].where(g["open"] > 0, g["close"])
    g["gap"] = o / g["prev_close_own"]
    g["gap_close"] = g["close"] / g["prev_close_own"]
    g["row"] = g.groupby("symbol", observed=True).cumcount().astype("int32")
    # did the stock also trade on the previous MARKET day? (False after a suspension / relisting)
    cal = np.sort(g["date"].unique())
    pos = np.searchsorted(cal, g["date"].to_numpy())
    prev_pos = g.groupby("symbol", observed=True)["date"].shift(1)
    pp = np.searchsorted(cal, prev_pos.fillna(g["date"]).to_numpy())
    g["consec"] = (pos - pp == 1) & prev_pos.notna().to_numpy()
    return g


def detect_actions(g: pd.DataFrame, ca: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (applied actions [symbol, adj_date, factor, method, ...], log of everything)."""
    grp_idx = g.groupby("symbol", observed=True).indices        # symbol -> row positions (no copies)
    applied, logrows = [], []
    ca = ca[ca["symbol"].isin(set(grp_idx))].copy()
    for (sym, ex), grp in ca.groupby(["symbol", "ex_date"]):
        s = g.iloc[grp_idx[sym]]
        idx = s.index[s["date"] >= ex]
        if len(idx) == 0:
            logrows.append({"symbol": sym, "ex_date": ex, "status": "no_trading_after_ex", "subject": grp["subject"].iloc[0]})
            continue
        first = s.loc[idx[0]]
        if (first["date"] - ex).days > 10:
            logrows.append({"symbol": sym, "ex_date": ex, "status": "no_trading_near_ex", "subject": grp["subject"].iloc[0]})
            continue
        cands = s.loc[idx[:3]]
        priced = grp[grp["factor"].notna()]
        done = False
        if len(priced):
            for _, row in cands.iterrows():
                for col in ["gap", "gap_close"]:
                    gv = row[col]
                    if not np.isfinite(gv):
                        continue
                    err = (gv / priced["factor"] - 1).abs()
                    best = err.idxmin()
                    if err[best] < TOL_CONFIRM:
                        applied.append({"symbol": sym, "adj_date": row["date"],
                                        "factor": float(priced.loc[best, "factor"]),
                                        "method": f"nse_confirmed_{col}", "kind": priced.loc[best, "kind"],
                                        "gap": float(gv), "ex_date": ex, "subject": priced.loc[best, "subject"]})
                        done = True
                        break
                if done:
                    break
        if not done and (grp["kind"] == "demerger").any():
            gv, gc = cands.iloc[0]["gap"], cands.iloc[0]["gap_close"]
            method = "demerger_gap"
            # old bhavcopies sometimes print a stale pre-open band price as the ex-date OPEN (ADANIENT
            # 2015-06-03: open = 0.90 x previous close, close = 0.17 x) -> use close-to-close then
            if np.isfinite(gc) and np.isfinite(gv) and gc < 0.8 * gv:
                gv, method = gc, "demerger_gap_close"
            if np.isfinite(gv) and gv < DEMERGER_MAX_GAP:
                applied.append({"symbol": sym, "adj_date": cands.iloc[0]["date"], "factor": float(gv),
                                "method": method, "kind": "demerger", "gap": float(gv),
                                "ex_date": ex, "subject": grp["subject"].iloc[0]})
                done = True
        if not done:
            logrows.append({"symbol": sym, "ex_date": ex, "status": "unconfirmed",
                            "gap_first": float(first["gap"]) if np.isfinite(first["gap"]) else None,
                            "subject": " | ".join(grp["subject"].astype(str).unique())[:200],
                            "factors": list(priced["factor"].round(4))})
    app = pd.DataFrame(applied)
    # ---- gap-only detection
    big = g[(g["gap"] < GAP_LO) | (g["gap"] > GAP_HI)].copy()
    big["symbol"] = big["symbol"].astype(str)
    if len(app):
        near_any = big.merge(app[["symbol", "adj_date"]], on="symbol", how="left")
        near_any = near_any[(near_any["adj_date"] - near_any["date"]).abs() <= pd.Timedelta(days=8)]
        explained = set(zip(near_any["symbol"], near_any["date"]))
        big = big[[(s, d_) not in explained for s, d_ in zip(big["symbol"], big["date"])]]
    # a day where many stocks gap without a record is a missing session, not corporate actions
    day_share = big.groupby("date").size() / g.groupby("date").size()
    mass_days = set(day_share[day_share > MASS_SHARE].index)
    for _, r in big.iterrows():
        if r["date"] in mass_days:
            logrows.append({"symbol": r["symbol"], "ex_date": r["date"], "status": "mass_gap_day_not_adjusted",
                            "gap_first": float(r["gap"])})
            continue
        ratios = np.array(STANDARD)
        err = np.abs(r["gap"] / ratios - 1)
        k = int(err.argmin())
        consec = bool(r["consec"]) if "consec" in r else True
        # only downward ratios (splits/bonuses) and only without a trading gap: an unexplained overnight
        # doubling, or a jump after a suspension, is treated as a real price move (D-006)
        if err[k] < SNAP_TOL and ratios[k] < 1 and consec:
            applied.append({"symbol": r["symbol"], "adj_date": r["date"], "factor": float(ratios[k]),
                            "method": "gap_snap", "kind": "unlisted", "gap": float(r["gap"]),
                            "ex_date": r["date"], "subject": ""})
        else:
            logrows.append({"symbol": r["symbol"], "ex_date": r["date"], "status": "unexplained_gap_not_adjusted",
                            "gap_first": float(r["gap"])})
    app = pd.DataFrame(applied)
    if len(app):
        app = app.sort_values(["symbol", "adj_date", "method"]).drop_duplicates(["symbol", "adj_date"])
    return app.reset_index(drop=True), pd.DataFrame(logrows)


def suffix_factors(actions: pd.DataFrame) -> pd.DataFrame:
    """Per symbol: for each adjustment date, product of that factor and all later ones."""
    a = actions.sort_values(["symbol", "adj_date"]).copy()
    a["suffix"] = a.groupby("symbol")["factor"].transform(lambda s: s[::-1].cumprod()[::-1])
    return a[["symbol", "adj_date", "suffix"]]


def cum_factor(df: pd.DataFrame, suffix: pd.DataFrame, date_col: str = "date") -> np.ndarray:
    """Price multiplier for each row = product of factors with adj_date > row date (else 1)."""
    if suffix.empty:
        return np.ones(len(df))
    left = df[[date_col, "symbol"]].copy()
    left["symbol"] = left["symbol"].astype(str)
    left["_i"] = np.arange(len(left))
    left = left.sort_values(date_col)
    right = suffix.rename(columns={"adj_date": date_col}).sort_values(date_col)
    m = pd.merge_asof(left, right, on=date_col, by="symbol", direction="forward", allow_exact_matches=False)
    out = np.ones(len(df))
    out[m["_i"].to_numpy()] = m["suffix"].fillna(1.0).to_numpy()
    return out


def main():
    ca = pd.read_parquet(REFERENCE / "corp_actions.parquet")
    g = gap_table()
    log(f"gap table: {len(g):,} rows, {g['symbol'].nunique()} symbols")
    actions, alog = detect_actions(g, ca)
    del g
    suffix = suffix_factors(actions) if len(actions) else pd.DataFrame(columns=["symbol", "adj_date", "suffix"])
    METRICS.mkdir(parents=True, exist_ok=True)
    actions.to_csv(METRICS / "corporate_actions_applied.csv", index=False)
    alog.to_csv(METRICS / "corporate_actions_not_applied.csv", index=False)
    actions.to_parquet(REFERENCE / "actions_applied.parquet", index=False)
    log(f"actions applied: {len(actions)} ({actions['method'].value_counts().to_dict() if len(actions) else {}}); "
        f"logged/not applied: {len(alog)} ({alog['status'].value_counts().to_dict() if len(alog) else {}})")
    # ---- equity
    root = PIT / "equity_adj"
    import shutil
    if root.exists():
        shutil.rmtree(root)
    big_moves = []
    years = sorted(int(p.name[5:]) for p in (PIT / "equity_raw").glob("year=*"))
    for y in years:
        e = pd.read_parquet(PIT / "equity_raw" / f"year={y}" / "part.parquet")
        e["symbol"] = e["symbol"].astype(str)
        cf = cum_factor(e, suffix)
        e["raw_close"] = e["close"]
        e["raw_volume"] = e["volume"]
        for c in PRICE_COLS:
            e[c] = e[c] * cf
        for c in QTY_COLS:
            e[c] = e[c] / cf
        e["cum_factor"] = cf
        e["symbol"] = e["symbol"].astype("category")
        p = root / f"year={y}" / "part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        e.to_parquet(p, index=False)
        del e
    # ---- F&O (OI in shares and futures prices)
    fo = read("fo_stock")
    fo["symbol"] = fo["symbol"].astype(str)
    cf = cum_factor(fo, suffix)
    for c in ["fut_oi", "fut_chg_oi", "near_oi", "call_oi", "put_oi"]:
        fo[c + "_adj"] = fo[c] / cf
    for c in ["near_settle", "near_close", "next_settle"]:
        fo[c + "_adj"] = fo[c] * cf
    fo["cum_factor"] = cf
    fo["symbol"] = fo["symbol"].astype("category")
    fo.to_parquet(PIT / "fo_stock_adj.parquet", index=False)
    # ---- QC: remaining extreme daily moves after adjustment (EQ chain)
    ext_parts, last = [], {}
    for p_ in sorted((PIT / "equity_adj").glob("year=*/part.parquet")):
        e = pd.read_parquet(p_, columns=["date", "symbol", "close"])
        e["symbol"] = e["symbol"].astype(str)
        e = e.sort_values(["symbol", "date"])
        prev = e.groupby("symbol")["close"].shift(1)
        first = prev.isna()
        prev[first] = e.loc[first, "symbol"].map(last)
        last.update(e.groupby("symbol")["close"].last().to_dict())
        e["ret"] = e["close"] / prev - 1
        ext_parts.append(e[e["ret"].abs() > 0.30])
    ext = pd.concat(ext_parts).sort_values("ret")
    qc = {"actions_applied": len(actions),
          "methods": actions["method"].value_counts().to_dict() if len(actions) else {},
          "not_applied": alog["status"].value_counts().to_dict() if len(alog) else {},
          "remaining_abs_moves_gt_30pct": int(len(ext)),
          "examples_remaining": ext.head(15)[["date", "symbol", "ret"]].astype(str).to_dict("records")}
    save_json(qc, METRICS / "adjustment_qc.json")
    log(f"remaining |daily move| > 30% after adjustment: {len(ext)}")


if __name__ == "__main__":
    main()
