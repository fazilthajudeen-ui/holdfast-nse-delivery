"""P0 exit check - data-quality report over the raw downloads and the PIT store.

    python -m impi.pipeline.p0_qc

Checks: expected row counts per day, gap detection (missing trading days / missed sessions),
sanity ranges, duplicates, delivery % cross-check against the bhavcopy's own quantities, legacy vs
sec_bhav agreement on overlap dates, F&O legacy vs UDiFF agreement, participant OI balance,
ban / VIX / index coverage. Writes results/metrics/qc_report.json and qc_report.md.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from impi.common import METRICS, RAW, config, d as as_date, log, save_json
from impi.ingest import nse_bhav, nse_fo
from impi.ingest.download import NO_DATA, candidate_dates, raw_path
from impi.transforms import pit_store as pit


def main():
    qc: dict = {"generated": dt.datetime.now().isoformat(timespec="seconds")}
    c = config()["data"]
    start, end = as_date(c["start"]), as_date(c["end"])
    cand = candidate_dates(start, end)
    nod = pd.read_csv(NO_DATA, header=None, names=["date", "dataset", "reason"]) if NO_DATA.exists() else pd.DataFrame()
    holidays = set(nod.loc[nod.dataset == "equity", "date"]) if len(nod) else set()
    trading = pit.trading_dates()
    unresolved = [x.isoformat() for x in cand if x.isoformat() not in holidays and x not in set(trading)
                  and x.weekday() < 5]
    qc["calendar"] = {"candidate_dates": len(cand), "trading_days": len(trading), "holidays": len(holidays),
                      "unresolved_weekdays": unresolved[:50], "n_unresolved": len(unresolved),
                      "first": str(trading[0]), "last": str(trading[-1]),
                      "trading_days_per_year": pd.Series([x.year for x in trading]).value_counts().sort_index().to_dict()}
    if len(nod):
        qc["missing_by_dataset"] = nod[nod.dataset != "equity"].groupby("dataset").size().to_dict()
        qc["stale_files_rejected"] = int(nod["reason"].astype(str).str.startswith("stale").sum())

    keys = ["deliv_pct_outside_0_100", "high_lt_low", "close_outside_high_low", "nonpositive_close",
            "negative_volume", "deliv_qty_gt_volume", "deliv_pct_vs_qty_mismatch_gt_0.05pt", "deliv_pct_checked"]
    rng_bad = dict.fromkeys(keys, 0)
    per_day_parts, coverage, sources, suspects = [], {}, {}, {}
    n_rows, dups, last_close = 0, 0, {}
    for p in sorted((pit.PIT / "equity_raw").glob("year=*/part.parquet")):
        y = int(p.parent.name[5:])
        eq = pd.read_parquet(p, columns=["date", "symbol", "series", "high", "low", "close", "volume",
                                         "deliv_qty", "deliv_pct", "prev_close", "source"])
        eq["symbol"] = eq["symbol"].astype(str)
        n_rows += len(eq)
        dups += int(eq.duplicated(["symbol", "date"]).sum())
        E = eq[eq["series"] == "EQ"]
        per_day_parts.append(E.groupby("date").size())
        rng_bad["deliv_pct_outside_0_100"] += int(((E["deliv_pct"] < 0) | (E["deliv_pct"] > 100)).sum())
        rng_bad["high_lt_low"] += int((E["high"] < E["low"]).sum())
        rng_bad["close_outside_high_low"] += int(((E["close"] > E["high"] * 1.0001) | (E["close"] < E["low"] * 0.9999)).sum())
        rng_bad["nonpositive_close"] += int((E["close"] <= 0).sum())
        rng_bad["negative_volume"] += int((E["volume"] < 0).sum())
        rng_bad["deliv_qty_gt_volume"] += int((E["deliv_qty"] > E["volume"]).sum())
        calc = E["deliv_qty"] / E["volume"] * 100
        both = calc.notna() & E["deliv_pct"].notna() & (E["volume"] > 0)
        rng_bad["deliv_pct_vs_qty_mismatch_gt_0.05pt"] += int(((calc - E["deliv_pct"]).abs()[both] > 0.05).sum())
        rng_bad["deliv_pct_checked"] += int(both.sum())
        coverage[y] = round(float(E["deliv_pct"].notna().mean()), 4)
        sources[y] = E["source"].astype(str).value_counts().to_dict()
        # missed-session detector: share of stocks whose file PREV_CLOSE != our previous close
        E2 = eq.sort_values(["symbol", "date"])
        prevc = E2.groupby("symbol")["close"].shift(1)
        first = prevc.isna()
        prevc[first] = E2.loc[first, "symbol"].map(last_close)
        last_close.update(E2.groupby("symbol")["close"].last().to_dict())
        sub = E2["series"] == "EQ"
        mism = ((E2["prev_close"] - prevc).abs() > 0.011 * prevc)[sub & prevc.notna()].groupby(E2.loc[sub & prevc.notna(), "date"]).mean()
        suspects.update({str(k.date()): round(float(v), 3) for k, v in mism[mism > 0.05].items()})
        del eq, E, E2
    per_day = pd.concat(per_day_parts).sort_index()
    med = per_day.rolling(60, min_periods=10, center=True).median()
    low_days = per_day[per_day < 0.7 * med]
    qc["rows"] = {"equity_rows": n_rows, "eq_rows_per_day_median": float(per_day.median()),
                  "days_with_<70pct_rows": {str(k.date()): int(v) for k, v in low_days.items()},
                  "duplicates_symbol_date": dups}
    qc["sanity"] = rng_bad
    qc["delivery_coverage_by_year"] = coverage
    qc["source_by_year"] = sources
    qc["missed_session_suspects"] = suspects

    # legacy vs sec_bhav overlap
    ov = []
    for x in c["overlap_check_dates"]:
        x = as_date(x)
        s, cm, mt = raw_path("sec_bhav", x), raw_path("cm_legacy", x), raw_path("mto", x)
        if not (s.exists() and cm.exists() and mt.exists()):
            ov.append({"date": str(x), "status": "files missing"})
            continue
        a = nse_bhav.parse_sec(s.read_bytes())
        b = nse_bhav.merge_legacy(nse_bhav.parse_cm(cm.read_bytes()), nse_bhav.parse_mto(mt.read_bytes()))
        m = a.merge(b, on=["symbol", "series"], suffixes=("_s", "_l"))
        res = {"date": str(x), "matched": len(m), "sec_rows": len(a), "legacy_rows": len(b)}
        for col in ["close", "volume", "deliv_qty"]:
            ok = m[col + "_s"].notna() & m[col + "_l"].notna()
            res[f"{col}_mismatch"] = int((~np.isclose(m.loc[ok, col + "_s"], m.loc[ok, col + "_l"])).sum())
        ov.append(res)
    qc["legacy_vs_sec_overlap"] = ov
    fo_ov = []
    for x in c["fo_overlap_check_dates"]:
        x = as_date(x)
        L, U = raw_path("fo_legacy", x), raw_path("fo_udiff", x)
        if not (L.exists() and U.exists()):
            fo_ov.append({"date": str(x), "status": "files missing"})
            continue
        a = nse_fo.aggregate_stock(nse_fo.parse_legacy(L.read_bytes()))
        b = nse_fo.aggregate_stock(nse_fo.parse_udiff(U.read_bytes()))
        m = a.merge(b, on=["date", "symbol"], suffixes=("_l", "_u"))
        fo_ov.append({"date": str(x), "underlyings": len(m),
                      "fut_oi_mismatch": int((~np.isclose(m.fut_oi_l, m.fut_oi_u)).sum()),
                      "call_oi_mismatch": int((~np.isclose(m.call_oi_l.fillna(-1), m.call_oi_u.fillna(-1))).sum())})
    qc["fo_legacy_vs_udiff"] = fo_ov

    fo = pit.read("fo_stock", columns=["date", "symbol", "fut_oi", "call_oi", "put_oi", "source"])
    qc["fo"] = {"rows": int(len(fo)), "underlyings_per_day_by_year": fo.assign(y=fo.date.dt.year).groupby("y")
                .apply(lambda g: round(float(g.groupby("date").size().median()), 1)).to_dict(),
                "negative_oi": int((fo["fut_oi"] < 0).sum()), "zero_fut_oi_rows": int((fo["fut_oi"] == 0).sum()),
                "trading_days_without_fo": len(set(trading) - set(fo["date"].dt.date))}
    po = pit.read("participant_oi")
    tot = po[po.participant == "TOTAL"]
    qc["participant_oi"] = {"days": int(po["date"].nunique()),
                            "total_long_ne_short": int((tot["fut_idx_long"] != tot["fut_idx_short"]).sum()
                                                       + (tot["fut_stk_long"] != tot["fut_stk_short"]).sum()),
                            "sum_of_participants_ne_total": int(sum(
                                (po[po.participant != "TOTAL"].groupby("date")[col].sum()
                                 != tot.set_index("date")[col]).sum() for col in ["fut_idx_long", "fut_stk_short"]))}
    ban = pit.read("ban")
    idx = pit.read("index")
    qc["ban"] = {"days_with_file": int(len(set(trading)) - len([x for x in trading if not raw_path("ban", x).exists()])),
                 "rows": int(len(ban)), "max_per_day": int(ban.groupby("date").size().max()) if len(ban) else 0}
    qc["index"] = {k: int(v) for k, v in idx.groupby("index").size().items()}
    vix = idx[idx["index"] == "india_vix"]
    qc["index"]["vix_first"] = str(vix["date"].min().date()) if len(vix) else None
    rep = json.loads((pit.PIT / "build_report.json").read_text()) if (pit.PIT / "build_report.json").exists() else {}
    qc["pit_build"] = {k: v for k, v in rep.items() if k in ("reference",)}
    save_json(qc, METRICS / "qc_report.json")
    lines = ["# P0 data-quality report", "", f"Generated {qc['generated']}", ""]
    for k, v in qc.items():
        if k == "generated":
            continue
        lines += [f"## {k}", "", "```", json.dumps(v, indent=1, default=str)[:4000], "```", ""]
    (METRICS / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"QC written: trading days {len(trading)}, unresolved weekdays {len(unresolved)}, sanity {rng_bad}")


if __name__ == "__main__":
    main()
