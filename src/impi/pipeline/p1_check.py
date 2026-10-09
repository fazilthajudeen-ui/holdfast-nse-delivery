"""P1 exit check - G1 (no look-ahead) test on REAL data.

    python -m impi.pipeline.p1_check

1. Every PIT table: available_at <= trade date + decision point (ban list: published the previous day).
2. Truncation invariance: for a seeded sample of F&O stocks, all stock-level features and VCP patterns
   computed on data cut at date D equal those computed on the full history, for every date <= D.
3. Market features (participant COT, regime) truncated vs full.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.common import METRICS, config, log, rng, save_json
from impi.features.compression import compression_features
from impi.features.delivery import delivery_features
from impi.features.positioning import market_positioning, stock_positioning
from impi.features.vcp import VCPDetector
from impi.features.volume import volume_features
from impi.transforms import pit_store as pit


def _stock_feats(df, fcfg):
    return pd.concat([delivery_features(df, fcfg), volume_features(df, fcfg), compression_features(df, fcfg),
                      stock_positioning(df, fcfg)], axis=1)


def main():
    fcfg = config()["features"]
    res = {"available_at": {}, "truncation": [], "market_truncation": {}}
    dec = pit.decision_offset()
    for name in ["equity_raw", "fo_stock"]:
        for p in sorted((pit.PIT / name).glob("year=*/part.parquet")):
            t = pd.read_parquet(p, columns=["date", "available_at"])
            bad = int((t["available_at"] > t["date"] + dec).sum())
            res["available_at"][name] = res["available_at"].get(name, 0) + bad
    for name in ["participant_oi", "index", "breadth"]:
        t = pit.read(name, columns=["date", "available_at"])
        res["available_at"][name] = int((t["available_at"] > t["date"] + dec).sum())
    b = pit.read("ban", columns=["date", "available_at"])
    res["available_at"]["ban_published_before_trade_date"] = int((b["available_at"] >= b["date"]).sum() == 0)
    fo = pd.read_parquet(pit.PIT / "fo_stock_adj.parquet")
    fo["symbol"] = fo["symbol"].astype(str)
    syms = sorted(fo["symbol"].unique())
    pick = list(rng(7).choice(syms, size=min(20, len(syms)), replace=False))
    eq = pit.read("equity_adj", columns=["date", "symbol", "open", "high", "low", "close", "raw_close", "volume",
                                         "deliv_pct", "turnover_rs"], symbols=pick)
    eq["symbol"] = eq["symbol"].astype(str)
    max_abs = 0.0
    for s in pick:
        df = eq[eq.symbol == s].sort_values("date").reset_index(drop=True)
        if len(df) < 400:
            continue
        f = fo[fo.symbol == s][["date", "fut_oi_adj", "call_oi_adj", "put_oi_adj", "near_settle", "near_expiry",
                                "next_settle", "next_expiry"]].rename(columns={"call_oi_adj": "call_oi", "put_oi_adj": "put_oi"})
        df = df.merge(f, on="date", how="left")
        full = _stock_feats(df, fcfg)
        det_full = VCPDetector(df.high, df.low, df.close, df.volume, fcfg["vcp"])
        for cut in rng(len(s)).integers(300, len(df) - 1, size=3):
            part = _stock_feats(df.iloc[:cut].copy(), fcfg)
            a, b2 = part.to_numpy(float), full.iloc[:cut].to_numpy(float)
            same_nan = np.array_equal(np.isnan(a), np.isnan(b2))
            diff = np.nanmax(np.abs(a - b2)) if np.isfinite(a).any() else 0.0
            max_abs = max(max_abs, float(diff))
            det_part = VCPDetector(df.high[:cut], df.low[:cut], df.close[:cut], df.volume[:cut], fcfg["vcp"])
            vcp_ok = all(det_part.pattern_at(t) == det_full.pattern_at(t) for t in range(cut - 60, cut))
            res["truncation"].append({"symbol": s, "cut": int(cut), "nan_pattern_equal": bool(same_nan),
                                      "max_abs_diff": float(diff), "vcp_equal": bool(vcp_ok)})
    po = pit.read("participant_oi")
    full = market_positioning(po, fcfg).set_index("date")
    cutd = po["date"].sort_values().unique()[len(po["date"].unique()) // 2]
    part = market_positioning(po[po.date <= cutd], fcfg).set_index("date")
    a, b2 = part.to_numpy(float), full.loc[part.index].to_numpy(float)
    res["market_truncation"] = {"cut": str(pd.Timestamp(cutd).date()),
                                "max_abs_diff": float(np.nanmax(np.abs(a - b2))),
                                "nan_pattern_equal": bool(np.array_equal(np.isnan(a), np.isnan(b2)))}
    ok = (all(v == 0 for k, v in res["available_at"].items() if k != "ban_published_before_trade_date")
          and res["available_at"]["ban_published_before_trade_date"] == 1
          and all(r["nan_pattern_equal"] and r["max_abs_diff"] < 1e-9 and r["vcp_equal"] for r in res["truncation"])
          and res["market_truncation"]["max_abs_diff"] < 1e-9 and res["market_truncation"]["nan_pattern_equal"])
    res["G1_PASS"] = bool(ok)
    res["n_checks"] = len(res["truncation"])
    save_json(res, METRICS / "p1_g1_check.json")
    log(f"G1 check: {'PASS' if ok else 'FAIL'} ({len(res['truncation'])} stock truncations, max diff {max_abs:.2e})")
    if not ok:
        raise SystemExit("G1 look-ahead check FAILED - see results/metrics/p1_g1_check.json")


if __name__ == "__main__":
    main()
