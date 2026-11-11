"""POST-HOC task A: placebo on the TEST set with 1,000 within-day delivery shuffles.

Same design as the P7 placebo (which used train+validation and 20 draws): delivery % is shuffled across
the test-event stocks within each trading day (EQ rows), the frozen delivery_features() recomputes
deliv_z20_prev, and the H1 exposed-minus-rest success-rate difference is recomputed on the frozen
events and labels. Warm-up from 2021-06-01 (>= 272 sessions before 2023-01-01); the unshuffled
recomputation is checked against the frozen values.

    .venv/Scripts/python.exe scripts/posthoc/s01_placebo.py [draws]      (~3.4 s per draw)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from _common import FIGDIR, mpl, save
from impi.common import config
from impi.features.delivery import delivery_features
from impi.pipeline.common_phase import tier1
from impi.pipeline.p2_events import load_events
from impi.transforms import pit_store as pit

DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
SEED = 20261111


def h1_diff(d):
    d = d[d["deliv_z20_prev"].notna() & d["label_10"].notna()]
    e = d["deliv_z20_prev"] > 0
    return float(d.loc[e, "label_10"].mean() - d.loc[~e, "label_10"].mean()) if e.any() and (~e).any() else np.nan


def main():
    t0 = time.time()
    ev = load_events()
    fams = {"BROAD": tier1(ev, "broad", ["test"]), "STRICT": tier1(ev, "strict", ["test"])}
    fcfg = config()["features"]
    syms = sorted(set(pd.concat(fams.values())["symbol"].astype(str)))
    warm = pd.Timestamp("2021-06-01")
    eq = pit.read("equity_adj", columns=["date", "symbol", "series", "close", "deliv_pct"], symbols=syms, start=warm)
    eq = eq[eq["date"] >= warm]
    eq["symbol"] = eq["symbol"].astype(str)
    eq = eq.sort_values(["symbol", "date"]).reset_index(drop=True)
    groups = [(s, g.index.to_numpy()) for s, g in eq.groupby("symbol", sort=False)]
    dates = eq["date"].to_numpy()
    idx_eq = np.flatnonzero((eq["series"] == "EQ").to_numpy())
    by_day = [ix.to_numpy() for _, ix in pd.Series(idx_eq).groupby(dates[idx_eq])]

    def one(dp):
        x = eq.copy()
        x["deliv_pct"] = dp
        z = []
        for s, ix in groups:
            g = x.loc[ix].reset_index(drop=True)
            z.append(pd.DataFrame({"symbol": s, "date": g["date"].to_numpy(),
                                   "dz": delivery_features(g, fcfg)["deliv_z20_prev"].to_numpy()}))
        z = pd.concat(z)
        return {k: h1_diff(v.drop(columns=["deliv_z20_prev"]).merge(z.rename(columns={"dz": "deliv_z20_prev"}),
                                                                    on=["symbol", "date"], how="left"))
                for k, v in fams.items()}

    actual = {k: h1_diff(v) for k, v in fams.items()}
    check = one(eq["deliv_pct"].to_numpy())
    r = np.random.default_rng(SEED)
    sims = {k: [] for k in fams}
    base = eq["deliv_pct"].to_numpy()
    for i in range(DRAWS):
        dp = base.copy()
        for ix in by_day:
            dp[ix] = base[r.permutation(ix)]
        res = one(dp)
        for k in fams:
            sims[k].append(res[k])
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{DRAWS} draws, {time.time() - t0:.0f}s", flush=True)
    out = {"task": "A placebo (test set)", "draws": DRAWS, "seed": SEED, "seconds": round(time.time() - t0, 1),
           "design": __doc__.strip().splitlines()[2:7], "unshuffled_recompute_equals_frozen": check,
           "frozen_actual": actual}
    for k in fams:
        a = np.array(sims[k], float)
        a = a[np.isfinite(a)]
        act = actual[k]
        out[k] = {"actual_diff_pp": 100 * act, "placebo_mean_pp": 100 * a.mean(), "placebo_sd_pp": 100 * a.std(ddof=1),
                  "n_valid": int(len(a)),
                  "p_one_sided_placebo_ge_actual": float((np.sum(a >= act) + 1) / (len(a) + 1)),
                  "p_two_sided": float((np.sum(np.abs(a - a.mean()) >= abs(act - a.mean())) + 1) / (len(a) + 1)),
                  "percentile_of_actual": float(100 * np.mean(a < act)),
                  "draws_pp": list(100 * a)}
    save(out, "A_placebo.json")
    plot(out)
    print("done", out["seconds"], "s")


def plot(out: dict):
    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.9))
    for ax, k in zip(axes, ["BROAD", "STRICT"]):
        a = np.array(out[k]["draws_pp"])
        ax.hist(a, bins=40, color="0.75", edgecolor="0.45", linewidth=0.4)
        ax.axvline(out[k]["actual_diff_pp"], color="C3", lw=1.4, label=f"actual {out[k]['actual_diff_pp']:+.2f} pp")
        ax.axvline(a.mean(), color="0.2", lw=0.8, ls="--", label=f"placebo mean {a.mean():+.2f} pp")
        ax.set_title(f"{k}: {len(a)} shuffles, one-sided p = {out[k]['p_one_sided_placebo_ge_actual']:.3f}", fontsize=7)
        ax.set_ylabel("draws")
        ax.legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)          # head-room for the legend
    axes[-1].set_xlabel("H1: exposed − rest success rate (pp)   [POST-HOC]")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(FIGDIR / "fig_placebo_hist.pdf")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        from _common import load
        plot(load("A_placebo.json"))
    else:
        main()
