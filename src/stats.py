"""Statistics for the pilot study -> data/derived/stats.json (+ robustness.parquet).

* Per tercile and horizon: n, mean/median excess return, bootstrap 95% CI of the mean.
* High vs Low: difference in mean excess return, Welch t-test, bootstrap 95% CI of the
  difference (10,000 resamples). Groups with n < min_n are labelled "insufficient data".
* Round-trip rate by tercile + chi-square test (3x2 and High-vs-Low 2x2).
* Robustness: re-detect events for every (move %, window) pair and re-test High vs Low.

    python -m src.stats
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as st

from src.common import load_config, log, path
from src.events import build_events, load_market


def bootstrap_mean_ci(x: np.ndarray, rng, n_boot: int, alpha=0.05):
    if len(x) < 2:
        return (np.nan, np.nan)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int, alpha=0.05):
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)
    return tuple(np.quantile(da - db, [alpha / 2, 1 - alpha / 2]))


def compare_high_low(ev: pd.DataFrame, col: str, rng, n_boot: int, min_n: int) -> dict:
    hi = ev.loc[ev["group"] == "High", col].dropna().to_numpy()
    lo = ev.loc[ev["group"] == "Low", col].dropna().to_numpy()
    out = {"n_high": len(hi), "n_low": len(lo),
           "mean_high": float(np.mean(hi)) if len(hi) else None,
           "mean_low": float(np.mean(lo)) if len(lo) else None}
    if len(hi) >= 2 and len(lo) >= 2:
        t, p = st.ttest_ind(hi, lo, equal_var=False)
        ci = bootstrap_diff_ci(hi, lo, rng, n_boot)
        out.update(diff=float(hi.mean() - lo.mean()), t=float(t), p=float(p),
                   ci_low=float(ci[0]), ci_high=float(ci[1]))
    else:
        out.update(diff=None, t=None, p=None, ci_low=None, ci_high=None)
    out["label"] = "insufficient data" if min(len(hi), len(lo)) < min_n else "ok"
    return out


def group_table(ev: pd.DataFrame, horizons, rng, n_boot, min_n) -> list[dict]:
    rows = []
    for g in ["Low", "Mid", "High"]:
        x = ev[ev["group"] == g]
        for h in horizons:
            v = x[f"exc_{h}"].dropna().to_numpy()
            ci = bootstrap_mean_ci(v, rng, n_boot)
            rows.append({"group": g, "horizon": h, "n": len(v),
                         "mean_excess": float(v.mean()) if len(v) else None,
                         "median_excess": float(np.median(v)) if len(v) else None,
                         "mean_raw": float(x[f"ret_{h}"].dropna().mean()) if len(v) else None,
                         "ci_low": float(ci[0]) if len(v) >= 2 else None,
                         "ci_high": float(ci[1]) if len(v) >= 2 else None,
                         "hit_rate": float((v > 0).mean() * 100) if len(v) else None,
                         "label": "insufficient data" if len(v) < min_n else "ok"})
    return rows


def roundtrip_table(ev: pd.DataFrame, min_n: int) -> dict:
    rows = []
    for g in ["Low", "Mid", "High"]:
        x = ev.loc[ev["group"] == g, "round_trip"].dropna()
        dd = ev.loc[(ev["group"] == g) & (ev["status"] == "complete"), "max_dd_60"]
        rows.append({"group": g, "n": int(len(x)), "round_trips": int(x.sum()),
                     "rate": float(x.mean() * 100) if len(x) else None,
                     "mean_max_dd_60": float(dd.mean()) if len(dd) else None,
                     "label": "insufficient data" if len(x) < min_n else "ok"})
    table = np.array([[r["round_trips"], r["n"] - r["round_trips"]] for r in rows])
    out = {"rows": rows, "chi2_3x2": None, "p_3x2": None, "chi2_high_low": None, "p_high_low": None}
    if (table.sum(axis=1) > 0).all() and (table.sum(axis=0) > 0).all():
        chi2, p, *_ = st.chi2_contingency(table)
        out.update(chi2_3x2=float(chi2), p_3x2=float(p))
        hl = table[[2, 0]]
        if (hl.sum(axis=0) > 0).all():
            chi2, p, *_ = st.chi2_contingency(hl)
            out.update(chi2_high_low=float(chi2), p_high_low=float(p))
    return out


def analyse(ev: pd.DataFrame, cfg: dict, rng) -> dict:
    s = cfg["stats"]
    horizons = cfg["events"]["horizons"]
    return {
        "n_events": int(len(ev)), "n_stocks": int(ev["symbol"].nunique()),
        "n_complete_60": int((ev["status"] == "complete").sum()),
        "groups": group_table(ev, horizons, rng, s["bootstrap_resamples"], s["min_n"]),
        "high_vs_low": {str(h): compare_high_low(ev, f"exc_{h}", rng, s["bootstrap_resamples"], s["min_n"])
                        for h in horizons},
        "round_trip": roundtrip_table(ev, s["min_n"]),
        "cutoffs": ev.groupby("group")["event_rel_deliv"].agg(["min", "max"]).round(2).to_dict("index"),
    }


def robustness(mkt, cfg: dict, base_diff: float | None, rng) -> pd.DataFrame:
    s, e, r = cfg["stats"], cfg["events"], cfg["robustness"]
    h = s["headline_horizon"]
    rows = []
    for mp in r["move_pcts"]:
        for w in r["windows"]:
            ev = build_events(mkt, mp, w, e["cooldown"], directions=("up",), cfg=cfg)
            res = compare_high_low(ev, f"exc_{h}", rng, 2000, s["min_n"]) if len(ev) else {}
            rt = roundtrip_table(ev, s["min_n"])["rows"] if len(ev) else []
            diff = res.get("diff")
            rows.append({
                "move_pct": int(round(mp * 100)), "window": w, "n_events": len(ev),
                "n_high": res.get("n_high", 0), "n_low": res.get("n_low", 0),
                f"diff_exc_{h}": diff, "p_value": res.get("p"),
                "rt_rate_high": rt[2]["rate"] if rt else None, "rt_rate_low": rt[0]["rate"] if rt else None,
                "same_direction": (None if diff is None or base_diff is None
                                   else bool(np.sign(diff) == np.sign(base_diff))),
                "significant_5pct": None if res.get("p") is None else bool(res["p"] < 0.05),
                "label": res.get("label", "insufficient data"),
                "is_baseline": mp == e["move_pct"] and w == e["window"],
            })
    return pd.DataFrame(rows)


def _json_default(o):
    if isinstance(o, np.generic):
        o = o.item()
        return None if isinstance(o, float) and np.isnan(o) else o
    if o is pd.NA or o is pd.NaT:
        return None
    return str(o)


def headline_text(res: dict, h: int) -> str:
    c = res["high_vs_low"][str(h)]
    if c["diff"] is None:
        return "Not enough completed events yet to compare High vs Low conviction."
    verb = "beat" if c["diff"] > 0 else "trailed"
    txt = (f"High-conviction up-moves {verb} low-conviction up-moves by {abs(c['diff']):.1f} "
           f"percentage points of excess return over {h} trading days "
           f"(n = {c['n_high']} vs {c['n_low']}, p = {c['p']:.3f}, "
           f"95% CI {c['ci_low']:+.1f} to {c['ci_high']:+.1f}).")
    if c["label"] != "ok":
        txt += " Insufficient data (n < 30 in a group): indicative only."
    elif c["p"] >= 0.05:
        txt += " The difference is not statistically significant at 5%."
    return txt


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg["stats"]["seed"])
    ev = pd.read_parquet(path("events"))
    mkt = load_market(cfg=cfg)
    h = cfg["stats"]["headline_horizon"]
    out = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
           "data_start": str(mkt.dates.min().date()), "data_end": str(mkt.dates.max().date()),
           "analysis_start": str(mkt.analysis_start.date()), "benchmark": mkt.bench_name,
           "n_stocks_universe": len(mkt.symbols), "params": cfg["events"], "headline_horizon": h}
    for d in ["up", "down"]:
        out[d] = analyse(ev[ev["direction"] == d], cfg, rng)
    out["headline"] = headline_text(out["up"], h)

    base = out["up"]["high_vs_low"][str(h)]["diff"]
    log("robustness grid (re-detecting events for each parameter pair)...")
    rob = robustness(mkt, cfg, base, rng)
    rob.to_parquet(path("events_all_params"), index=False)
    out["robustness"] = rob.to_dict("records")
    comparable = rob[~rob["is_baseline"] & rob["same_direction"].notna()]
    out["robustness_holds"] = f"{int(comparable['same_direction'].sum())} of {len(comparable)}"

    with open(path("stats"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)

    print("RESULTS (up-moves)")
    print("  " + out["headline"])
    g = pd.DataFrame(out["up"]["groups"])
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print(g.pivot(index="group", columns="horizon", values=["n", "mean_excess"]).loc[["Low", "Mid", "High"]])
        print("\n  High vs Low by horizon:")
        for hh, c in out["up"]["high_vs_low"].items():
            if c["diff"] is not None:
                print(f"    +{hh:>2}d: diff {c['diff']:+.2f} pp, p {c['p']:.3f}, "
                      f"CI [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}], n {c['n_high']}/{c['n_low']} ({c['label']})")
        rt = out["up"]["round_trip"]
        print("\n  Round-trip rate: " + ", ".join(
            f"{r['group']} {r['rate']:.0f}% (n={r['n']})" for r in rt["rows"] if r["rate"] is not None)
            + (f"; chi-square p = {rt['p_3x2']:.3f}" if rt["p_3x2"] is not None else ""))
        d = out["down"]["high_vs_low"][str(h)]
        if d["diff"] is not None:
            print(f"\nRESULTS (down-moves, secondary): High - Low excess {h}d = {d['diff']:+.2f} pp, "
                  f"p {d['p']:.3f}, n {d['n_high']}/{d['n_low']}")
        print(f"\nROBUSTNESS (direction holds in {out['robustness_holds']} alternative settings)")
        print(rob.drop(columns=["is_baseline"]).to_string(index=False))
    log(f"saved {path('stats').name}")


if __name__ == "__main__":
    main()
