"""Parse cached bhavcopies -> adjusted, filtered daily panel (data/clean/panel.parquet).

Steps
1. Parse every raw CSV: strip column names and values, "-" -> NaN, dates -> datetime.
2. Build one price chain per symbol from EQ/BE/BZ rows (a stock can hop between series;
   NSE's PREV_CLOSE refers to the previous close in whichever series it traded).
3. Corporate actions: factor = PREV_CLOSE[t] / CLOSE[t-1]. If |factor - 1| > threshold the
   day is a split/bonus: all earlier prices are multiplied by the factor and quantities
   divided by it. Every adjustment is logged to data/logs/corporate_actions.csv.
4. Keep SERIES == EQ rows only (BE/T2T are delivery-only by rule -> fake 100% delivery).
5. Universe: Nifty 500 list (latest dated copy), else all EQ stocks; then drop stocks with
   < min_trading_days rows or median turnover < ₹1 crore.

    python -m src.clean
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import load_config, log, path

NUMERIC = ["PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "LAST_PRICE", "CLOSE_PRICE",
           "AVG_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS", "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"]
RENAME = {"SYMBOL": "symbol", "SERIES": "series", "DATE1": "date", "PREV_CLOSE": "prev_close",
          "OPEN_PRICE": "open", "HIGH_PRICE": "high", "LOW_PRICE": "low", "LAST_PRICE": "last",
          "CLOSE_PRICE": "close", "AVG_PRICE": "avg_price", "TTL_TRD_QNTY": "volume",
          "TURNOVER_LACS": "turnover_lacs", "NO_OF_TRADES": "trades", "DELIV_QTY": "deliv_qty",
          "DELIV_PER": "deliv_per_nse"}
PRICE_COLS = ["prev_close", "open", "high", "low", "last", "close", "avg_price"]
QTY_COLS = ["volume", "deliv_qty"]


# ---------------------------------------------------------------- parsing
def parse_bhav(src: str | Path | bytes) -> pd.DataFrame:
    """Parse one sec_bhavdata_full file (path, raw bytes or text)."""
    if isinstance(src, bytes):
        src = io.StringIO(src.decode("utf-8", errors="replace"))
    elif isinstance(src, str) and "\n" in src:
        src = io.StringIO(src)
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda s: s.str.strip())
    df = df.replace({"-": np.nan, "": np.nan})
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["DATE1"] = pd.to_datetime(df["DATE1"], format="%d-%b-%Y")
    return df.rename(columns=RENAME)


def load_raw() -> pd.DataFrame:
    files = sorted(path("raw").glob("sec_bhavdata_full_*.csv"))
    if not files:
        raise SystemExit("No raw files. Run: python -m src.download --months 14")
    log(f"parsing {len(files)} raw files")
    raw = pd.concat([parse_bhav(f) for f in files], ignore_index=True)
    dup = raw.duplicated(["symbol", "series", "date"])
    if dup.any():
        log(f"dropping {dup.sum()} duplicate rows (same symbol/series/date in two files)")
    return raw[~dup].reset_index(drop=True)


def load_benchmark() -> pd.DataFrame:
    """Nifty 500 closing value per day from the cached ind_close_all files."""
    rows = []
    for f in sorted(path("index").glob("ind_close_all_*.csv")):
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        hit = df[df["Index Name"].str.strip().str.upper() == "NIFTY 500"]
        if len(hit):
            rows.append({"date": pd.to_datetime(hit["Index Date"].iloc[0], format="%d-%m-%Y"),
                         "nifty500": float(hit["Closing Index Value"].iloc[0])})
    return pd.DataFrame(rows, columns=["date", "nifty500"])


# ---------------------------------------------------------------- corporate actions
# Standard split / bonus price ratios (new price / old price), with snap tolerance.
# split 1:2 -> 1/2, 1:5 -> 1/5, 1:10 -> 1/10, FV 5->2 -> 2/5; bonus 1:1 -> 1/2, 2:1 -> 1/3,
# 1:2 -> 2/3, 1:3 -> 3/4, 3:2 -> 2/5; consolidations (reverse splits) -> 2, 5, 10.
STANDARD_RATIOS = {1 / 2: .06, 1 / 3: .06, 1 / 4: .06, 1 / 5: .06, 1 / 10: .06, 1 / 20: .06, 2 / 5: .05,
                   2 / 3: .04, 3 / 4: .03, 3 / 5: .04, 2.0: .06, 5.0: .06, 10.0: .06}


def snap_ratio(gap: float) -> float | None:
    """Return the standard split/bonus ratio `gap` is close to, else None."""
    if not np.isfinite(gap) or 0.72 < gap < 1.5:
        return None
    for r, tol in STANDARD_RATIOS.items():
        if abs(gap / r - 1) <= tol:
            return r
    return None


def adjust_corporate_actions(df: pd.DataFrame, threshold: float,
                             mass_event_share: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Back-adjust prices/quantities per symbol. `df` = one row per (symbol, date), any series.

    Detection (either rule):
      A. PREV_CLOSE[t] / CLOSE[t-1] differs from 1 by > threshold (NSE adjusted PREV_CLOSE).
      B. OPEN[t] / CLOSE[t-1] is < 0.72 or > 1.5 AND within a few % of a standard split/bonus
         ratio -> factor snapped to that ratio. Needed because sec_bhavdata_full usually
         carries the *unadjusted* previous close on ex-dates.
    If more than `mass_event_share` of symbols trigger rule A on the same day, the day follows
    a missing trading session instead: nothing is adjusted and a warning is logged.
    Returns (adjusted df with `adj_mult` column, log of detected actions).
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    prev_raw_close = g["close"].shift(1)
    factor_a = df["prev_close"] / prev_raw_close
    is_a = (factor_a - 1).abs() > threshold
    if mass_event_share is not None and df["symbol"].nunique() >= 50:
        share = is_a.groupby(df["date"]).mean()
        mass = share[share > mass_event_share]
        for d, s in mass.items():
            log(f"WARNING {d:%Y-%m-%d}: {s:.0%} of stocks show a PREV_CLOSE jump -> a trading session "
                f"before this date is missing from data/raw (add it to download.extra_dates). Not adjusted.")
        is_a &= ~df["date"].isin(mass.index)
    gap = (df["open"] / prev_raw_close).to_numpy()
    snapped = pd.Series([snap_ratio(x) for x in gap], index=df.index, dtype="float64")
    # C. extreme gap (< 0.5x or > 2x) that is not a standard ratio: demerger / capital
    #    restructuring (e.g. Vedanta 2026). Use the actual open/prev-close ratio.
    gap_s = pd.Series(gap, index=df.index)
    extreme = ((gap_s < 0.5) | (gap_s > 2.0)) & snapped.isna()
    snapped = snapped.where(~extreme, gap_s)
    is_b = snapped.notna() & ~is_a
    factor = factor_a.where(is_a, snapped)
    is_ca = is_a | is_b
    f = factor.where(is_ca, 1.0).fillna(1.0)
    # multiplier for day t = product of factors of all actions strictly after t
    rev_cum = f.iloc[::-1].groupby(df["symbol"].iloc[::-1]).cumprod().iloc[::-1]
    df["adj_mult"] = rev_cum.groupby(df["symbol"]).shift(-1).fillna(1.0)
    for c in PRICE_COLS:
        df[c] = df[c] * df["adj_mult"]
    for c in QTY_COLS:
        df[c] = df[c] / df["adj_mult"]
    actions = pd.DataFrame({
        "symbol": df.loc[is_ca, "symbol"], "ex_date": df.loc[is_ca, "date"].dt.date,
        "series": df.loc[is_ca, "series"], "prev_close_nse": df.loc[is_ca, "prev_close"] / df.loc[is_ca, "adj_mult"],
        "last_close": prev_raw_close[is_ca], "open": df.loc[is_ca, "open"] / df.loc[is_ca, "adj_mult"],
        "factor": factor[is_ca].round(6),
        "method": np.where(is_a[is_ca], "prev_close",
                           np.where(extreme[is_ca], "open_gap_unsnapped", "open_gap_snapped")),
    })
    actions["implied_ratio"] = (1 / actions["factor"]).round(3)   # e.g. 2.0 for a 1:2 split
    return df, actions


def pick_chain_rows(raw: pd.DataFrame, chain: list[str]) -> pd.DataFrame:
    """One row per (symbol, date): EQ preferred, then the next series in `chain`."""
    x = raw[raw["series"].isin(chain)].copy()
    x["_rank"] = x["series"].map({s: i for i, s in enumerate(chain)})
    x = x.sort_values(["symbol", "date", "_rank"]).drop_duplicates(["symbol", "date"])
    return x.drop(columns="_rank")


def universe_symbols() -> tuple[set[str] | None, str]:
    cfg = load_config()["universe"]
    if cfg["mode"] != "nifty500":
        return None, "all EQ stocks (liquidity filter)"
    lists = sorted(path("universe").glob("ind_nifty500list_*.csv"))
    if not lists:
        return None, "all EQ stocks (Nifty 500 list unavailable)"
    df = pd.read_csv(lists[-1])
    df.columns = [c.strip() for c in df.columns]
    return set(df["Symbol"].str.strip()), f"Nifty 500 ({lists[-1].name})"


# ---------------------------------------------------------------- panel
def build_panel(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = load_config()
    ucfg = cfg["universe"]
    chain = pick_chain_rows(raw, ucfg["series_chain"])
    ca = cfg["corporate_actions"]
    adj, actions = adjust_corporate_actions(chain, ca["threshold"], ca.get("mass_event_share"))
    # Price continuity keeps BE/BZ days (a stock parked in BE for a week would otherwise
    # show a fake jump), but only EQ rows carry delivery: BE/T2T is delivery-only by rule
    # and would fake 100% delivery.
    eq = adj[adj["symbol"].isin(adj.loc[adj["series"].isin(ucfg["series_keep"]), "symbol"])].copy()
    non_eq = ~eq["series"].isin(ucfg["series_keep"])
    eq.loc[non_eq, ["deliv_qty", "deliv_per_nse"]] = np.nan

    info = {"raw_rows": len(raw), "eq_symbols": eq["symbol"].nunique(), "non_eq_rows": int(non_eq.sum())}
    members, info["universe"] = universe_symbols()
    if members is not None:
        info["universe_listed"] = len(members)
        eq = eq[eq["symbol"].isin(members)]
        info["universe_found"] = eq["symbol"].nunique()

    stats = eq.groupby("symbol").agg(days=("date", "size"), med_turn=("turnover_lacs", "median"))
    keep = stats[(stats["days"] >= ucfg["min_trading_days"])
                 & (stats["med_turn"] >= ucfg["min_median_turnover_lacs"])].index
    info["dropped_short_history"] = int((stats["days"] < ucfg["min_trading_days"]).sum())
    info["dropped_low_turnover"] = int(((stats["days"] >= ucfg["min_trading_days"])
                                        & (stats["med_turn"] < ucfg["min_median_turnover_lacs"])).sum())
    panel = eq[eq["symbol"].isin(keep)].reset_index(drop=True)
    actions = actions[actions["symbol"].isin(keep)].reset_index(drop=True)
    info["symbols"] = panel["symbol"].nunique()
    info["non_eq_rows"] = int((~panel["series"].isin(ucfg["series_keep"])).sum())
    return panel, actions, info


def quality_report(panel: pd.DataFrame, info: dict) -> str:
    per_day = panel.groupby("date").size()
    dates = per_day.index
    gaps = pd.Series(dates[1:] - dates[:-1], index=dates[1:])
    big_gaps = gaps[gaps > pd.Timedelta(days=4)]
    miss_deliv = panel["deliv_qty"].isna().mean() * 100
    lines = [
        "DATA QUALITY REPORT (clean panel)",
        f"  universe            : {info['universe']}",
        f"  raw rows parsed     : {info['raw_rows']:,}",
        f"  EQ symbols in raw   : {info['eq_symbols']:,}",
    ]
    if "universe_found" in info:
        lines.append(f"  universe listed/found: {info['universe_listed']} / {info['universe_found']}")
    lines += [
        f"  dropped (<{load_config()['universe']['min_trading_days']} days) : {info['dropped_short_history']}",
        f"  dropped (turnover < ₹1 cr): {info['dropped_low_turnover']}",
        f"  symbols kept        : {info['symbols']}",
        f"  trading days        : {len(dates)}  ({dates.min():%Y-%m-%d} -> {dates.max():%Y-%m-%d})",
        f"  rows per day        : min {per_day.min()}, median {int(per_day.median())}, max {per_day.max()}",
        f"  panel rows          : {len(panel):,}",
        f"  missing delivery    : {miss_deliv:.2f}% of rows "
        f"({info['non_eq_rows']} BE/BZ-series rows kept for price only, delivery blanked)",
        f"  corporate actions   : {info.get('corporate_actions', 0)} (see data/logs/corporate_actions.csv)",
        f"  gaps > 4 calendar days: {len(big_gaps)}"
        + ("".join(f"\n      {d:%Y-%m-%d} (gap {g.days} d)" for d, g in big_gaps.items())),
    ]
    low_days = per_day[per_day < 0.8 * per_day.median()]
    if len(low_days):
        lines.append(f"  days with <80% of usual rows: {', '.join(f'{d:%Y-%m-%d}' for d in low_days.index[:10])}")
    return "\n".join(lines)


def main():
    raw = load_raw()
    panel, actions, info = build_panel(raw)
    info["corporate_actions"] = len(actions)
    actions.to_csv(path("logs") / "corporate_actions.csv", index=False)
    panel.to_parquet(path("clean"), index=False)
    bench = load_benchmark()
    bench.to_parquet(path("benchmark"), index=False)
    report = quality_report(panel, info)
    report += f"\n  Nifty 500 index days: {len(bench)}"
    print(report)
    (path("logs") / "data_quality_clean.txt").write_text(report, encoding="utf-8")
    if len(actions):
        print("\nCorporate actions detected (first 15):")
        print(actions.head(15).to_string(index=False))
    log(f"saved {path('clean').relative_to(path('clean').parents[2])}")


if __name__ == "__main__":
    main()
