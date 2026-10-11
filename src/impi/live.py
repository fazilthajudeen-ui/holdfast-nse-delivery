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
RESEARCH_MODELS_V3 = ROOT / "data" / "processed" / "models" / "v3_models.pkl"
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
    if RESEARCH_MODELS_V3.exists():
        _run("impi.v3.options", env)
        _run("impi.v3.panel", env)


# ---------------------------------------------------------------- scoring + export (in-process, live env)
def export() -> dict:
    import numpy as np
    import pandas as pd

    from impi.common import PROCESSED, save_json
    from impi.transforms import pit_store as pit
    from impi.v2.train import add_xs_ranks, probability, score

    v2 = pickle.loads(RESEARCH_MODELS.read_bytes())
    v3 = pickle.loads(RESEARCH_MODELS_V3.read_bytes()) if RESEARCH_MODELS_V3.exists() else {}
    adopt = (_load_metric("v3_models.json").get("adoption") or {}) if v3 else {}
    if v3:
        from impi.v3.train import add_xs_ranks as add_xs_ranks_v3
        panel = pd.read_parquet(PROCESSED / "features" / "panel_v3.parquet")
    else:
        panel = pd.read_parquet(PROCESSED / "features" / "panel_t1.parquet")
    as_of = panel["date"].max()
    hist_from = as_of - pd.Timedelta(days=400)
    recent = panel[panel["date"] >= hist_from].copy()
    recent = add_xs_ranks_v3(recent) if v3 else add_xs_ranks(recent)
    for t, m in v2.items():
        recent[f"raw2_{t}"] = score(m, recent)
        recent[f"p2_{t}"] = probability(m, recent[f"raw2_{t}"].to_numpy())
    for t, m in v3.items():
        recent[f"raw3_{t}"] = score(m, recent)
        recent[f"p3_{t}"] = probability(m, recent[f"raw3_{t}"].to_numpy())
    # the model shown per target: v3 where the pre-registered adoption rule says so (or a new target), else v2
    models, shown = {}, {}
    for t in list(v2) + [t for t in v3 if t not in v2]:
        use3 = t in v3 and (t not in v2 or (adopt.get(t) or {}).get("adopt"))
        models[t], shown[t] = (v3[t], "v3") if use3 else (v2[t], "v2")
        k = "3" if use3 else "2"
        recent[f"raw_{t}"], recent[f"p_{t}"] = recent[f"raw{k}_{t}"], recent[f"p{k}_{t}"]
    # within-day percentiles of the raw scores (isotonic probabilities are step functions, so many stocks
    # can share one probability when a model separates them only weakly)
    pct = {"xs_pct": "label_xs10", "breakout_pct": "label_10", "vol_pct": "vol_exp", "xs20_pct": "label_xs20",
           "breakout20_pct": "label_20"}
    for col, t in pct.items():
        if f"raw_{t}" in recent:
            recent[col] = recent.groupby("date")[f"raw_{t}"].rank(pct=True) * 100
    today = recent[recent["date"] == as_of].copy()
    # per-stock explanation: XGBoost feature contributions for each shown model
    expl = {}
    for t in models:
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
            "turnover_z20", "iv30", "iv_rv", "iv_term", "iv_skew", "iv_pct252", "iv_chg5", "log_pcr_vol"]
    rows = []
    for r in today.itertuples(index=False):
        d = r._asdict()
        row = {"symbol": d["symbol"], "sector": sectors.get(d["symbol"], "Unknown"), "close": d["raw_close"],
               "p_breakout": d["p_label_10"], "p_outperform": d["p_label_xs10"], "p_vol_expansion": d["p_vol_exp"],
               "p_outperform20": d.get("p_label_xs20"), "p_breakout20": d.get("p_label_20"),
               "xs_pct": d["xs_pct"], "breakout_pct": d["breakout_pct"], "vol_pct": d["vol_pct"],
               "xs20_pct": d.get("xs20_pct"), "breakout20_pct": d.get("breakout20_pct"),
               "events": today_ev.get(d["symbol"], []), "explain": expl.get(d["symbol"], {})}
        row.update({k: d.get(k) for k in keep})
        rows.append(row)
    APP.mkdir(parents=True, exist_ok=True)
    hist = recent[["date", "symbol"] + [c for c in ["p_label_10", "p_label_xs10", "p_vol_exp", "xs_pct", "p_label_xs20",
                                                    "p_label_20"] if c in recent]].copy()
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
    # the v2 track record always logs the v2 models (unchanged since 9 Oct 2026), whatever the terminal shows
    today_v2 = today.copy()
    for t in v2:
        today_v2[f"p_{t}"] = today_v2[f"p2_{t}"]
    today_v2["xs_pct"] = today_v2.groupby("date")["raw2_label_xs10"].rank(pct=True) * 100
    track = update_track(as_of, today_v2, panel)
    track_v3 = update_track_v3(as_of, today, panel) if v3 else None
    research = research_summary(shown)
    meta = {"as_of": as_of.strftime("%Y-%m-%d"), "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "n_stocks": int(len(today)), "events_today": {f: int(sum(f in v for v in today_ev.values())) for f in ["all", "broad", "strict"]},
            "models": research.get("models", {}), "disclaimer": DISCLAIMER, "track": track, "track_v3": track_v3}
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


V3_TARGETS = ["vol_exp", "label_xs10", "label_10", "label_xs20", "label_20"]
FORWARD_MIN_DATES = 60          # pre_registered_v3.md: evaluation once 60 prediction dates have matured (20-day targets)


def _date_bootstrap(d, stat, n_boot=2000, seed=20261011):
    """One-sided p-value that stat > 0, resampling prediction DATES with replacement (pre_registered_v3.md)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby("date")]
    draws = []
    for _ in range(n_boot):
        s = stat(pd_concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
        if s is not None:
            draws.append(s)
    draws = np.asarray(draws)
    return float((np.sum(draws <= 0) + 1) / (len(draws) + 1)) if len(draws) else None


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def forward_tests(j) -> dict:
    """F1-F5 of pre_registered_v3.md on matured v3 predictions (raw scores rank the stocks)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from impi.validation.multiple_testing import benjamini_hochberg

    def auc(d, s, y):
        return float(roc_auc_score(d[y].astype(int), d[s])) if d[y].nunique() == 2 else None

    def minus_half(s, y):
        return lambda d: (lambda a: None if a is None else a - 0.5)(auc(d, s, y))

    tests = {}
    m = j[j["vol_exp"].notna() & j["raw2_vol_exp"].notna()]
    if len(m) >= 50:
        diff = lambda d: (lambda a, b: None if a is None or b is None else a - b)(auc(d, "raw3_vol_exp", "vol_exp"),
                                                                                   auc(d, "raw2_vol_exp", "vol_exp"))
        tests["F1"] = {"claim": "v3 vol model AUC > v2", "n": int(len(m)), "dates": int(m["date"].nunique()),
                       "stat": diff(m), "p": _date_bootstrap(m, diff)}
    for key, t in [("F2", "vol_exp"), ("F3", "label_xs20"), ("F4", "label_20"), ("F5", "label_xs10")]:
        m = j[j[t].notna()]
        if len(m) >= 50 and m[t].nunique() == 2:
            tests[key] = {"claim": f"v3 {t} AUC > 0.5", "n": int(len(m)), "dates": int(m["date"].nunique()),
                          "auc": auc(m, f"raw3_{t}", t), "p": _date_bootstrap(m, minus_half(f"raw3_{t}", t))}
    keys = [k for k in tests if tests[k]["p"] is not None]
    if keys:
        q, _ = benjamini_hochberg(np.array([tests[k]["p"] for k in keys]), 0.10)
        for k, qq in zip(keys, q):
            tests[k]["bh_q"] = float(qq)
            tests[k]["supported"] = bool(qq <= 0.10)
    return tests


def update_track_v3(as_of, today, panel) -> dict:
    """v3 forward track record (pre_registered_v3.md): log raw scores + probabilities of all v3 targets and the v2
    volatility score (for the paired test F1) BEFORE outcomes exist; evaluate F1-F5 once 60 dates have matured."""
    import pandas as pd

    TRACK.mkdir(parents=True, exist_ok=True)
    f = TRACK / "predictions_v3.csv"
    log = pd.read_csv(f, parse_dates=["date"]) if f.exists() else pd.DataFrame()
    if log.empty or not (log["date"] == as_of).any():
        cols = ["date", "symbol", "raw2_vol_exp"] + [f"{k}3_{t}" for t in V3_TARGETS for k in ("raw", "p")]
        new = today[cols].copy()
        new["logged_at"] = dt.datetime.now().isoformat(timespec="seconds")
        log = pd.concat([log, new], ignore_index=True)
        log.to_csv(f, index=False)
    lab = panel[["date", "symbol"] + V3_TARGETS]
    j = log.merge(lab, on=["date", "symbol"], how="left")
    out = {"first_logged": str(log["date"].min().date()), "days_logged": int(log["date"].nunique()),
           "predictions": int(len(log)), "matured": {t: int(j[t].notna().sum()) for t in V3_TARGETS}}
    dates20 = int(j.loc[j["label_xs20"].notna(), "date"].nunique())
    out["matured_dates_20d"] = dates20
    out["status"] = "final" if dates20 >= FORWARD_MIN_DATES else f"interim ({dates20}/{FORWARD_MIN_DATES} dates matured; descriptive only)"
    out["tests"] = forward_tests(j)
    (TRACK / "summary_v3.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _load_metric(name: str) -> dict:
    p = RESEARCH_METRICS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def research_summary(shown: dict | None = None) -> dict:
    load = _load_metric
    p6, v2, p2, v3 = load("p6_test.json"), load("v2_models.json"), load("p2_summary.json"), load("v3_models.json")
    conf = [{k: t.get(k) for k in ["hypothesis", "family", "n", "diff", "p", "bh_q", "expected_sign", "supported"]}
            for t in p6.get("confirmatory", [])]
    models = {}
    for t, ch in (v2.get("chosen") or {}).items():
        ph = (v2.get("posthoc_test") or {}).get(t, {})
        models[t] = {"feature_set": ch.get("feature_set"), "model": ch.get("model"), "cv_auc": ch.get("mean_auc"),
                     "cv_fold_auc": ch.get("fold_auc"), "posthoc_test_auc": ph.get("auc"), "posthoc_base_rate": ph.get("base_rate"),
                     "posthoc_rate_by_decile": ph.get("rate_by_decile"), "daily_ic": ph.get("daily_ic"),
                     "top_decile_net": ph.get("top_decile_minus_all_net_of_costs"), "version": "v2"}
    for t, ch in (v3.get("chosen") or {}).items():
        if (shown or {}).get(t) != "v3":
            continue
        ph = (v3.get("posthoc_test") or {}).get(t, {})
        models[t] = {"feature_set": ch.get("feature_set"), "model": ch.get("model"), "cv_auc": ch.get("mean_auc"),
                     "cv_fold_auc": ch.get("fold_auc"), "posthoc_test_auc": ph.get("auc"), "posthoc_base_rate": ph.get("base_rate"),
                     "posthoc_rate_by_decile": ph.get("rate_by_decile"), "daily_ic": ph.get("daily_ic"),
                     "top_decile_net": ph.get("top_decile_minus_all_net_of_costs"), "version": "v3"}
    v3_summary = None
    if v3:
        v3_summary = {"chosen": v3.get("chosen"), "best_without_options": v3.get("best_without_options"),
                      "adoption": v3.get("adoption"), "posthoc_test": v3.get("posthoc_test"), "shown": shown}
    return {"v1_confirmatory": conf, "v1_ml_test": p6.get("ml_test"), "v1_backtests": p6.get("backtests"),
            "v1_spa": p6.get("spa"), "v1_walk_forward": p6.get("walk_forward"), "events": p2.get("tier1_eligible_counts"),
            "models": models, "v2_broad": (v2.get("posthoc_test") or {}).get("broad_events_label_10_auc"),
            "v2_label": (v2.get("posthoc_test") or {}).get("label"), "v3": v3_summary}


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
