"""Synthetic point-in-time store for an end-to-end REHEARSAL of P1-P7 (no real data, no real results).

    IMPI_SANDBOX=1 IMPI_PROCESSED_DIR=<dir>/processed IMPI_RESULTS_DIR=<dir>/results \
        python tests/integration/make_synthetic.py

60 random-walk stocks 2014-2026 (40 with F&O), participant OI, indices, breadth, bans, results dates.
A weak true signal is planted (higher delivery before a rise) so the tests have something to find.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.common import PROCESSED, SANDBOX
from impi.transforms import pit_store as pit

assert SANDBOX, "refusing to write synthetic data outside sandbox mode"


def main(seed: int = 7):
    r = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", "2026-10-09")
    n = len(dates)
    rows, fo_rows = [], []
    for k in range(60):
        sym = f"SYN{k:02d}"
        deliv = np.clip(45 + np.cumsum(r.normal(0, 1.5, n)) * 0.2 + r.normal(0, 6, n), 5, 95)
        drift = 0.0004 + 0.0008 * (np.convolve(deliv - 45, np.ones(20) / 20, mode="same") / 20)   # planted signal
        ret = drift + r.normal(0, 0.018, n)
        # occasional quiet stretches (compression) followed by breakouts
        quiet = (np.sin(np.arange(n) / (37 + k)) > 0.7)
        ret = np.where(quiet, ret * 0.35, ret)
        close = 200 * np.exp(np.cumsum(ret))
        openp = close * np.exp(r.normal(0, 0.004, n))
        high = np.maximum(openp, close) * (1 + np.abs(r.normal(0, 0.007, n)))
        low = np.minimum(openp, close) * (1 - np.abs(r.normal(0, 0.007, n)))
        vol = r.lognormal(13, 0.35, n) * np.where(quiet, 0.7, 1.0)
        df = pd.DataFrame({"date": dates, "symbol": sym, "series": "EQ", "open": openp, "high": high, "low": low,
                           "close": close, "last": close, "prev_close": np.r_[close[0], close[:-1]],
                           "raw_close": close, "volume": vol, "deliv_qty": vol * deliv / 100, "deliv_pct": deliv,
                           "turnover_rs": vol * close, "trades": vol / 50, "source": "synthetic",
                           "cum_factor": 1.0, "raw_volume": vol})
        rows.append(df)
        if k < 40:
            oi = np.abs(2e6 + np.cumsum(r.normal(0, 3e4, n)))
            exp = dates + pd.offsets.BMonthEnd(0)
            fo_rows.append(pd.DataFrame({"date": dates, "symbol": sym, "fut_oi": oi, "fut_oi_adj": oi,
                                         "call_oi_adj": oi * 0.6, "put_oi_adj": oi * 0.45, "call_oi": oi * 0.6,
                                         "put_oi": oi * 0.45, "near_settle": close * 1.004, "near_expiry": exp,
                                         "next_settle": close * 1.009, "next_expiry": exp + pd.offsets.BMonthEnd(1)}))
    eq = pd.concat(rows, ignore_index=True)
    eq["available_at"] = eq["date"] + pd.Timedelta(hours=19, minutes=30)
    for name in ["equity_raw", "equity_adj"]:
        pit._write_by_year(eq, name)
    fo = pd.concat(fo_rows, ignore_index=True)
    fo["available_at"] = fo["date"] + pd.Timedelta(hours=19, minutes=30)
    pit._write_by_year(fo, "fo_stock")
    fo.to_parquet(pit.PIT / "fo_stock_adj.parquet", index=False)
    po = []
    walk = {p: np.cumsum(r.normal(0, 1, n)) for p in ["Client", "DII", "FII", "Pro"]}
    for i, d in enumerate(dates):
        tot = 1e6
        for p in ["Client", "DII", "FII", "Pro"]:
            x = 0.05 * np.tanh(walk[p][i] / 20)
            po.append({"date": d, "participant": p, "fut_idx_long": tot * (0.25 + x), "fut_idx_short": tot * (0.25 - x),
                       "fut_stk_long": tot * (0.25 - x / 2), "fut_stk_short": tot * (0.25 + x / 2)})
        po.append({"date": d, "participant": "TOTAL", "fut_idx_long": tot, "fut_idx_short": tot,
                   "fut_stk_long": tot, "fut_stk_short": tot})
    po = pd.DataFrame(po)
    po["available_at"] = po["date"] + pd.Timedelta(hours=20)
    po.to_parquet(pit.PIT / "participant_oi.parquet", index=False)
    mkt = eq.groupby("date")["close"].mean()
    idx = []
    for name, scale in [("nifty50", 1.0), ("nifty200", 0.5), ("nifty500", 0.4)]:
        c = mkt / mkt.iloc[0] * 8000 * scale
        idx.append(pd.DataFrame({"date": c.index, "index": name, "open": c.values * 0.999, "high": c.values * 1.005,
                                 "low": c.values * 0.995, "close": c.values}))
    vix = 15 + 5 * np.sin(np.arange(n) / 50) + r.normal(0, 1, n)
    idx.append(pd.DataFrame({"date": dates, "index": "india_vix", "open": vix, "high": vix, "low": vix, "close": vix}))
    idx = pd.concat(idx, ignore_index=True)
    idx["available_at"] = idx["date"] + pd.Timedelta(hours=18)
    idx.to_parquet(pit.PIT / "index.parquet", index=False)
    chg = eq.assign(c=eq["close"] / eq["prev_close"] - 1).groupby("date")["c"]
    br = pd.DataFrame({"date": dates, "advances": chg.apply(lambda s: int((s > 0).sum())).values,
                       "declines": chg.apply(lambda s: int((s < 0).sum())).values, "n": 60})
    br["available_at"] = br["date"] + pd.Timedelta(hours=18, minutes=30)
    br.to_parquet(pit.PIT / "breadth.parquet", index=False)
    ban = pd.DataFrame({"date": dates[::97], "symbol": [f"SYN{k % 40:02d}" for k in range(len(dates[::97]))]})
    ban["available_at"] = ban["date"] - pd.Timedelta(hours=4)
    ban.to_parquet(pit.PIT / "ban.parquet", index=False)
    pit.REFERENCE.mkdir(parents=True, exist_ok=True)
    res = []
    for k in range(60):
        for q in pd.date_range("2014-02-10", "2026-09-30", freq="QS"):
            md = q + pd.Timedelta(days=int(r.integers(30, 45)))
            res.append({"symbol": f"SYN{k:02d}", "meeting_date": md, "known_at": md - pd.Timedelta(days=7)})
    pd.DataFrame(res).to_parquet(pit.REFERENCE / "results_dates.parquet", index=False)
    pd.DataFrame({"symbol": [f"SYN{k:02d}" for k in range(60)],
                  "sector": [["Banks", "IT", "Pharma", "Auto"][k % 4] for k in range(60)]}).to_parquet(
        pit.REFERENCE / "sector_map.parquet", index=False)
    print("synthetic PIT written to", PROCESSED)


if __name__ == "__main__":
    main()
