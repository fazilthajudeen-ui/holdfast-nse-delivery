"""Point-in-time store: raw NSE files -> normalised Parquet tables with an `available_at` timestamp.

    python -m impi.transforms.pit_store            # (re)build every table from data/raw

Tables (data/processed/pit/<name>/year=YYYY/*.parquet unless noted):
  equity_raw      unadjusted daily rows, series EQ/BE/BZ, symbol-mapped, delivery blanked for non-EQ
  fo_stock        per-underlying daily F&O aggregates (futures OI in shares, near/next futures, option OI)
  participant_oi  market-wide participant OI (contracts)            -> participant_oi.parquet
  ban             (date, symbol) F&O ban list by trade date         -> ban.parquet
  index           nifty50/200/500 closes + India VIX                -> index.parquet
  breadth         advances/declines over all EQ stocks per day      -> breadth.parquet
  reference/      corp_actions, symbol_changes, results_dates, sector_map, symbol_map_log

`available_at` = trade date + release time from data/release_lag.yaml (G1). Memory-light: files are
parsed one at a time and written per year.
"""
from __future__ import annotations

import datetime as dt
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from impi.common import PROCESSED, RAW, config, log, save_json
from impi.ingest import corp_actions as CA
from impi.ingest import indices, nse_bhav, nse_fo, participant, surveillance
from impi.ingest.download import raw_path

PIT = PROCESSED / "pit"
REFERENCE = PIT / "reference"


# ------------------------------------------------------------------ release times (G1)
def release_offset(dataset: str) -> pd.Timedelta:
    """Offset from trade-date midnight to public release, e.g. 'T 18:30' -> 18h30m."""
    rel = config()["release_lag"]["datasets"][dataset]["release"]
    day, hhmm = rel.split()
    base = {"T": 0, "T-1": -1}[day]
    h, m = (int(x) for x in hhmm.split(":"))
    return pd.Timedelta(days=base, hours=h, minutes=m)


def decision_offset() -> pd.Timedelta:
    day, hhmm = config()["release_lag"]["decision_point"].split()
    h, m = (int(x) for x in hhmm.split(":"))
    return pd.Timedelta(hours=h, minutes=m)


def assert_usable_at_t(dataset: str) -> None:
    """G1 guard: a dataset may feed a day-T feature only if it is released by the T decision point."""
    if release_offset(dataset) > decision_offset():
        raise RuntimeError(f"G1 violation: {dataset} is released after the T decision point; lag it by one day")


# ------------------------------------------------------------------ helpers
def trading_dates() -> list[dt.date]:
    out = set()
    for name in ["sec_bhav", "cm_legacy"]:
        for p in (RAW / name).glob("*/*"):
            if p.suffix in (".csv", ".zip"):
                out.add(dt.date.fromisoformat(p.stem.rsplit("_", 1)[1]))
    end = config()["data"]["end"]
    return sorted(d for d in out if d <= end)


def _write_by_year(df: pd.DataFrame, name: str) -> None:
    root = PIT / name
    if root.exists():
        shutil.rmtree(root)
    for y, part in df.groupby(df["date"].dt.year):
        p = root / f"year={y}" / "part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(p, index=False)


def read(name: str, columns: list[str] | None = None, start=None, end=None, as_of=None,
         symbols=None) -> pd.DataFrame:
    """Read a PIT table. as_of (Timestamp) keeps only rows with available_at <= as_of (G1)."""
    single = PIT / f"{name}.parquet"
    if single.exists():
        df = pd.read_parquet(single, columns=columns)
    else:
        files = sorted((PIT / name).glob("year=*/part.parquet"))
        if start is not None:
            files = [f for f in files if int(f.parent.name[5:]) >= pd.Timestamp(start).year]
        if end is not None:
            files = [f for f in files if int(f.parent.name[5:]) <= pd.Timestamp(end).year]
        filt = [("symbol", "in", list(symbols))] if symbols is not None else None
        df = pd.concat([pd.read_parquet(f, columns=columns, filters=filt) for f in files], ignore_index=True)
    if start is not None and "date" in df:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None and "date" in df:
        df = df[df["date"] <= pd.Timestamp(end)]
    if as_of is not None:
        if "available_at" not in df:
            raise KeyError(f"{name} has no available_at column")
        df = df[df["available_at"] <= pd.Timestamp(as_of)]
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ equity
def _equity_for_date(d: dt.date) -> pd.DataFrame | None:
    sec = raw_path("sec_bhav", d)
    if sec.exists():
        df = nse_bhav.parse_sec(sec.read_bytes())
    else:
        cm = raw_path("cm_legacy", d)
        if not cm.exists():
            return None
        mto = raw_path("mto", d)
        df = nse_bhav.merge_legacy(nse_bhav.parse_cm(cm.read_bytes()),
                                   nse_bhav.parse_mto(mto.read_bytes()) if mto.exists() else None)
    return df


def build_equity(changes: pd.DataFrame) -> dict:
    """Parsed and written one calendar year at a time (low memory)."""
    chain = config()["universe"]["series_chain"]
    keep = config()["universe"]["series_keep"]
    dates = trading_dates()
    root = PIT / "equity_raw"
    if root.exists():
        shutil.rmtree(root)
    breadth, applied_all = [], []
    qc = {"dates": len(dates), "missing_delivery_days": [], "rows": 0, "symbols": set()}
    for year in sorted({d.year for d in dates}):
        frames = []
        for d in [x for x in dates if x.year == year]:
            df = _equity_for_date(d)
            if df is None:
                continue
            eqs = df[df["series"].isin(keep)]
            chg = eqs["close"] / eqs["prev_close"] - 1
            breadth.append({"date": pd.Timestamp(d), "advances": int((chg > 0).sum()),
                            "declines": int((chg < 0).sum()), "n": int(chg.notna().sum())})
            df = df[df["series"].isin(chain)].drop(columns=["isin"])
            if df.loc[df["series"].isin(keep), "deliv_qty"].isna().mean() > 0.5:
                qc["missing_delivery_days"].append(d.isoformat())
            frames.append(df)
        eq = pd.concat(frames, ignore_index=True)
        del frames
        # non-EQ series are delivery-only (trade-to-trade) -> 100% delivery would be fake
        eq.loc[~eq["series"].isin(keep), ["deliv_qty", "deliv_pct"]] = np.nan
        eq, applied = CA.apply_symbol_changes(eq, changes)
        applied_all += [dict(a, year=year) for a in applied if a["status"] != "applied"]
        # one row per symbol-day: prefer EQ when a symbol has two series on the same day
        eq["_pri"] = (~eq["series"].isin(keep)).astype(int)
        eq = eq.sort_values(["symbol", "date", "_pri"]).drop_duplicates(["symbol", "date"]).drop(columns="_pri")
        eq["available_at"] = eq["date"] + release_offset("delivery_mto")   # delivery is released last
        for c in ["symbol", "series", "source"]:
            eq[c] = eq[c].astype("category")
        p = root / f"year={year}" / "part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        eq.to_parquet(p, index=False)
        qc["rows"] += len(eq)
        qc["symbols"] |= set(eq["symbol"].astype(str).unique())
        log(f"  equity {year}: {len(eq):,} rows")
        del eq
    b = pd.DataFrame(breadth)
    b["available_at"] = b["date"] + release_offset("equity_bhav")
    b.to_parquet(PIT / "breadth.parquet", index=False)
    qc["symbols"] = len(qc["symbols"])
    qc["symbol_change_conflicts"] = applied_all[:200]
    return qc


# ------------------------------------------------------------------ F&O
def _fo_for_date(d: dt.date) -> pd.DataFrame | None:
    last_legacy = config()["data"]["sources"]["fo_legacy"]["last_date"]
    order = [("fo_legacy", nse_fo.parse_legacy), ("fo_udiff", nse_fo.parse_udiff)]
    if d > last_legacy:
        order.reverse()
    for name, parser in order:
        p = raw_path(name, d)
        if p.exists():
            agg = nse_fo.aggregate_stock(parser(p.read_bytes()))
            agg["source"] = name
            return agg
    return None


def build_fo(changes: pd.DataFrame, workers: int = 3) -> dict:
    dates = trading_dates()
    frames, missing = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for d, agg in zip(dates, ex.map(_fo_for_date, dates, chunksize=16)):
            if agg is None or agg.empty:
                missing.append(d.isoformat())
            else:
                frames.append(agg)
    fo = pd.concat(frames, ignore_index=True)
    fo, applied = CA.apply_symbol_changes(fo, changes)
    fo = fo.drop_duplicates(["symbol", "date"])
    fo["symbol"] = fo["symbol"].astype("category")
    fo["available_at"] = fo["date"] + release_offset("fo_bhav")
    _write_by_year(fo, "fo_stock")
    return {"rows": len(fo), "underlyings": int(fo["symbol"].nunique()), "missing_dates": missing,
            "per_day_median": float(fo.groupby("date").size().median())}


# ------------------------------------------------------------------ small tables
def build_small(changes: pd.DataFrame) -> dict:
    dates = trading_dates()
    po, ban, idx, miss = [], [], [], {"participant_oi": [], "ban": [], "index_close": []}
    for d in dates:
        p = raw_path("participant_oi", d)
        if p.exists():
            po.append(participant.parse(p.read_bytes(), d))
        else:
            miss["participant_oi"].append(d.isoformat())
        p = raw_path("ban", d)
        if p.exists():
            ban.append(surveillance.parse_ban(p.read_bytes(), d))
        else:
            miss["ban"].append(d.isoformat())
        p = raw_path("index_close", d)
        if p.exists():
            idx.append(indices.parse(p.read_bytes(), d))
        else:
            miss["index_close"].append(d.isoformat())
    po = pd.concat(po, ignore_index=True)
    po["available_at"] = po["date"] + release_offset("participant_oi")
    po.to_parquet(PIT / "participant_oi.parquet", index=False)
    ban = pd.concat(ban, ignore_index=True) if ban else pd.DataFrame(columns=["date", "symbol"])
    ban, _ = CA.apply_symbol_changes(ban, changes)
    # ban list for trade date D is published the previous trading day evening
    prev = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
    ban["available_at"] = [pd.Timestamp(prev.get(x.date(), x.date())) + release_offset("ban_list") + pd.Timedelta(days=1)
                           for x in ban["date"]]
    ban.to_parquet(PIT / "ban.parquet", index=False)
    idx = pd.concat(idx, ignore_index=True)
    vix_files = idx[idx["index"] == "india_vix"][["date", "close"]]
    vh = CA.load_vix_history()
    vh = vh[~vh["date"].isin(vix_files["date"])]
    vh["index"] = "india_vix"
    idx = pd.concat([idx, vh[["date", "index", "open", "high", "low", "close"]]], ignore_index=True)
    idx = idx[idx["date"] <= pd.Timestamp(config()["data"]["end"])]
    idx = idx.drop_duplicates(["date", "index"]).sort_values(["index", "date"])
    idx["available_at"] = idx["date"] + release_offset("index_close")
    idx.to_parquet(PIT / "index.parquet", index=False)
    return {"participant_rows": len(po), "ban_rows": len(ban), "index_rows": len(idx),
            "missing": {k: len(v) for k, v in miss.items()}, "missing_dates": miss,
            "vix_from_history_api": int(len(vh))}


def build_reference() -> tuple[pd.DataFrame, dict]:
    REFERENCE.mkdir(parents=True, exist_ok=True)
    changes = CA.load_symbol_changes()
    changes.to_parquet(REFERENCE / "symbol_changes.parquet", index=False)
    ca = CA.load_corp_actions()
    ca, _ = CA.apply_symbol_changes(ca.rename(columns={"ex_date": "date"}), changes)
    ca = ca.rename(columns={"date": "ex_date"})
    ca.to_parquet(REFERENCE / "corp_actions.parquet", index=False)
    rd = CA.load_results_dates()
    rd, _ = CA.apply_symbol_changes(rd.rename(columns={"meeting_date": "date"}), changes)
    rd = rd.rename(columns={"date": "meeting_date"})
    rd.to_parquet(REFERENCE / "results_dates.parquet", index=False)
    sm = CA.load_sector_map()
    sm.to_parquet(REFERENCE / "sector_map.parquet", index=False)
    return changes, {"symbol_changes": len(changes), "corp_actions": len(ca),
                     "corp_action_kinds": ca["kind"].value_counts().to_dict(),
                     "results_dates": len(rd), "sector_labels": len(sm)}


def main():
    PIT.mkdir(parents=True, exist_ok=True)
    qc = {"built": dt.datetime.now().isoformat(timespec="seconds")}
    changes, qc["reference"] = build_reference()
    log(f"reference: {qc['reference']}")
    qc["small"] = build_small(changes)
    log(f"participant/ban/index: { {k: v for k, v in qc['small'].items() if k != 'missing_dates'} }")
    qc["equity"] = build_equity(changes)
    log(f"equity: rows={qc['equity']['rows']:,} symbols={qc['equity']['symbols']}")
    qc["fo"] = build_fo(changes)
    log(f"fo: rows={qc['fo']['rows']:,} underlyings={qc['fo']['underlyings']}")
    save_json(qc, PIT / "build_report.json")


if __name__ == "__main__":
    main()
