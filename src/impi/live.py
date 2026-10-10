"""IMPI live mode: nightly update for the terminal + a prospective (forward) track record.

    python -m impi.live            # download new days, rebuild into data/live, score, export for the app
    python -m impi.live --no-download

Runs in a SEPARATE folder (data/live/...), with the data end extended to today, so the frozen research
results in data/processed and results/ are never touched. Uses the frozen v2 models trained on
2015-2022 only. Every evening's predictions are appended to data/track/predictions.csv BEFORE their
outcomes exist; matured predictions are scored in data/track/summary.json (the only truly
out-of-sample test left after the one-time research test).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "live"
TRACK = ROOT / "data" / "track"
APP = LIVE / "app"
RESEARCH_MODELS = ROOT / "data" / "processed" / "models" / "v2_models.pkl"
RESEARCH_METRICS = ROOT / "results" / "metrics"
DISCLAIMER = "Research tool for education - not investment advice."


def _env(end: dt.date) -> dict:
    env = dict(os.environ)
    env.update({"IMPI_PROCESSED_DIR": str(LIVE / "processed"), "IMPI_RESULTS_DIR": str(LIVE / "results"),
                "IMPI_DATA_END": end.isoformat()})
    return env


def _run(mod: str, env: dict, *args: str) -> None:
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", mod, *args], cwd=ROOT, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    print(f"[{dt.datetime.now():%H:%M:%S}] {mod} ({time.time() - t0:.0f}s): {' | '.join(tail)[-300:]}", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"{mod} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def rebuild(end: dt.date, download: bool) -> None:
    env = _env(end)
    if download:
        start = (end - dt.timedelta(days=10)).isoformat()
        _run("impi.ingest.download", env, "--start", start, "--end", end.isoformat())
    _run("impi.transforms.pit_store", env)
    _run("impi.transforms.adjust", env)
    _run("impi.features.build", env)
    _run("impi.v2.panel", env)


# ---------------------------------------------------------------- scoring + export (in-process, live env)
def export() -> dict:
    import numpy as np
    import pandas as pd

    from impi.common import PROCESSED, save_json
    from impi.transforms import pit_store as pit
    from impi.v2.train import add_xs_ranks, probability, score

    models = pickle.loads(RESEARCH_MODELS.read_bytes())
    panel = pd.read_parquet(PROCESSED / "features" / "panel_t1.parquet")
    as_of = panel["date"].max()
    hist_from = as_of - pd.Timedelta(days=400)
    recent = add_xs_ranks(panel[panel["date"] >= hist_from].copy())
    for t, m in models.items():
        recent[f"raw_{t}"] = score(m, recent)
        recent[f"p_{t}"] = probability(m, recent[f"raw_{t}"].to_numpy())
    recent["xs_pct"] = recent.groupby("date")["raw_label_xs10"].rank(pct=True) * 100
    # within-day percentiles of the raw scores (isotonic probabilities are step functions, so many stocks
    # can share one probability when a model separates them only weakly)
    recent["breakout_pct"] = recent.groupby("date")["raw_label_10"].rank(pct=True) * 100
    recent["vol_pct"] = recent.groupby("date")["raw_vol_exp"].rank(pct=True) * 100
    today = recent[recent["date"] == as_of].copy()
    # per-stock explanation: XGBoost feature contributions for today's breakout & relative models
    expl = {}
    for t in ["label_10", "label_xs10", "vol_exp"]:
        m = models[t]
        if "xgb" in m["parts"] or "ranker" in m["parts"]:
            import xgboost as xgb
            booster = (m["parts"].get("xgb") or m["parts"]["ranker"]).get_booster()
            contrib = booster.predict(xgb.DMatrix(today[m["cols"]].astype("float32"), feature_names=list(m["cols"])),
                                      pred_contribs=True)
            names = list(m["cols"])
            for i, sym in enumerate(today["symbol"]):
                c = contrib[i][:-1]
                order = np.argsort(-np.abs(c))[:5]
                expl.setdefault(sym, {})[t] = [{"feature": names[j].replace("xr_", ""), "contrib": round(float(c[j]), 4),
                                                "value": None if pd.isna(today.iloc[i][names[j]]) else round(float(today.iloc[i][names[j]]), 4)}
                                               for j in order]
    ev = pd.read_parquet(PROCESSED / "features" / "events.parquet", columns=["date", "symbol", "family", "tier1", "eligible"])
    ev = ev[ev["tier1"]]
    today_ev = ev[ev["date"] == as_of].groupby("symbol")["family"].apply(lambda s: sorted(set(s))).to_dict()
    sectors = pd.read_parquet(pit.REFERENCE / "sector_map.parquet").set_index("symbol")["sector"].to_dict()
    keep = ["deliv_pct", "deliv_z20", "deliv_z5", "vol_z", "bbw_pct252", "atr_ratio", "range_contraction", "vcp_valid", "vcp_n",
            "oi_chg5", "oi_chg20", "oi_cot", "long_buildup", "short_covering", "short_buildup", "long_unwinding",
            "basis_ann", "log_pcr_oi", "ret_1", "ret_5", "ret_20", "ret_60", "dist_52w_high", "rv20", "rv20_pct252",
            "turnover_z20"]
    rows = []
    for r in today.itertuples(index=False):
        d = r._asdict()
        row = {"symbol": d["symbol"], "sector": sectors.get(d["symbol"], "Unknown"), "close": d["raw_close"],
               "p_breakout": d["p_label_10"], "p_outperform": d["p_label_xs10"], "p_vol_expansion": d["p_vol_exp"],
               "xs_pct": d["xs_pct"], "breakout_pct": d["breakout_pct"], "vol_pct": d["vol_pct"],
               "events": today_ev.get(d["symbol"], []), "explain": expl.get(d["symbol"], {})}
        row.update({k: d.get(k) for k in keep})
        rows.append(row)
    APP.mkdir(parents=True, exist_ok=True)
    hist = recent[["date", "symbol", "p_label_10", "p_label_xs10", "p_vol_exp", "xs_pct"]].copy()
    hist.to_parquet(APP / "history.parquet", index=False)
    pd.DataFrame(rows).to_json(APP / "today.json", orient="records", date_format="iso", double_precision=6)
    ev_hist = ev[ev["date"] >= hist_from - pd.Timedelta(days=800)][["date", "symbol", "family"]]
    ev_hist.to_parquet(APP / "events.parquet", index=False)
    # market positioning + regime
    mk = pd.read_parquet(PROCESSED / "features" / "market.parquet")
    mk = mk[mk["date"] >= as_of - pd.Timedelta(days=3 * 365)]
    mcols = ["date", "nifty_above_200", "nifty_trend_50_200", "vix_pct252", "breadth10", "fii_flow20", "dii_flow20",
             "nifty500_ret20"] + [f"{p}_{s}_{k}" for p in ["fii", "dii", "pro", "client"] for s in ["fut_idx", "fut_stk"]
                                  for k in ["share", "cot"]]
    mk[[c for c in mcols if c in mk.columns]].to_parquet(APP / "market.parquet", index=False)
    idx = pit.read("index")
    idx = idx[idx["date"] >= as_of - pd.Timedelta(days=3 * 365)][["date", "index", "close"]]
    idx.to_parquet(APP / "index.parquet", index=False)
    track = update_track(as_of, today, panel)
    research = research_summary()
    meta = {"as_of": as_of.strftime("%Y-%m-%d"), "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "n_stocks": int(len(today)), "events_today": {f: int(sum(f in v for v in today_ev.values())) for f in ["all", "broad", "strict"]},
            "models": research.get("models", {}), "disclaimer": DISCLAIMER, "track": track}
    save_json(meta, APP / "meta.json")
    save_json(research, APP / "research.json")
    print(f"[{dt.datetime.now():%H:%M:%S}] export: as of {meta['as_of']}, {len(today)} stocks, events {meta['events_today']}", flush=True)
    return meta


def update_track(as_of, today, panel) -> dict:
    """Append today's predictions (once per date) and score every matured prediction."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    TRACK.mkdir(parents=True, exist_ok=True)
    f = TRACK / "predictions.csv"
    log = pd.read_csv(f, parse_dates=["date"]) if f.exists() else pd.DataFrame()
    if log.empty or not (log["date"] == as_of).any():
        new = today[["date", "symbol", "p_label_10", "p_label_xs10", "p_vol_exp", "xs_pct"]].copy()
        new["logged_at"] = dt.datetime.now().isoformat(timespec="seconds")
        log = pd.concat([log, new], ignore_index=True)
        log.to_csv(f, index=False)
    lab = panel[["date", "symbol", "label_10", "label_xs10", "vol_exp", "fwd_10"]]
    j = log.merge(lab, on=["date", "symbol"], how="left")
    out = {"first_logged": str(log["date"].min().date()) if len(log) else None, "days_logged": int(log["date"].nunique()),
           "predictions": int(len(log))}
    for t, p in [("label_10", "p_label_10"), ("label_xs10", "p_label_xs10"), ("vol_exp", "p_vol_exp")]:
        m = j[j[t].notna()]
        r = {"matured": int(len(m)), "days": int(m["date"].nunique())}
        if len(m) >= 50 and m[t].nunique() == 2:
            r["auc"] = float(roc_auc_score(m[t].astype(int), m[p]))
            r["base_rate"] = float(m[t].mean())
        out[t] = r
    m = j[j["fwd_10"].notna()]
    if len(m) >= 50:
        top = m[m["xs_pct"] >= 90].groupby("date")["fwd_10"].mean()
        allm = m.groupby("date")["fwd_10"].mean()
        out["top_decile_minus_all_fwd10"] = float((top - allm).mean())
    (TRACK / "summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def research_summary() -> dict:
    def load(name):
        p = RESEARCH_METRICS / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    p6, v2, p2 = load("p6_test.json"), load("v2_models.json"), load("p2_summary.json")
    conf = [{k: t.get(k) for k in ["hypothesis", "family", "n", "diff", "p", "bh_q", "expected_sign", "supported"]}
            for t in p6.get("confirmatory", [])]
    models = {}
    for t, ch in (v2.get("chosen") or {}).items():
        ph = (v2.get("posthoc_test") or {}).get(t, {})
        models[t] = {"feature_set": ch.get("feature_set"), "model": ch.get("model"), "cv_auc": ch.get("mean_auc"),
                     "cv_fold_auc": ch.get("fold_auc"), "posthoc_test_auc": ph.get("auc"), "posthoc_base_rate": ph.get("base_rate"),
                     "posthoc_rate_by_decile": ph.get("rate_by_decile"), "daily_ic": ph.get("daily_ic"),
                     "top_decile_net": ph.get("top_decile_minus_all_net_of_costs")}
    return {"v1_confirmatory": conf, "v1_ml_test": p6.get("ml_test"), "v1_backtests": p6.get("backtests"),
            "v1_spa": p6.get("spa"), "v1_walk_forward": p6.get("walk_forward"), "events": p2.get("tier1_eligible_counts"),
            "models": models, "v2_broad": (v2.get("posthoc_test") or {}).get("broad_events_label_10_auc"),
            "v2_label": (v2.get("posthoc_test") or {}).get("label")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--end", help="ISO date (default: today)")
    a = ap.parse_args(argv)
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today()
    if not RESEARCH_MODELS.exists():
        raise SystemExit("v2 models not found - run python -m impi.v2.train first")
    if not a.export_only:
        rebuild(end, download=not a.no_download)
    # export runs in a child with the live environment so every path points at data/live
    env = _env(end)
    r = subprocess.run([sys.executable, "-c", "from impi.live import export; export()"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-1500:], flush=True)
    if r.returncode != 0:
        raise SystemExit("export failed")


if __name__ == "__main__":
    main()
