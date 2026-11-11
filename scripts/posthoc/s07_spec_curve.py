"""POST-HOC task C: specification curve for H1 and H2 on the test set (every spec reported).

Grid
  barrier   : (+1.5, -1.0), (+2.0, -1.0) [frozen], (+3.0, -1.5) ATR14
  horizon   : 5, 10 [frozen], 20 trading days
  threshold : H1 delivery expansion deliv_z20_prev > 0 [frozen], > 0.5, > 1.0   (H2 has no delivery threshold)
  event set : BROAD [frozen confirmatory set], B1 all 20-day-high breakouts
H1: 3 x 3 x 3 x 2 = 54 specs; H2: 3 x 3 x 2 = 18 specs. Labels are recomputed with the frozen
triple_barrier(); an event is in the test split only if its new label window ends by 2026-10-09 (same purge
rule). Test = two-proportion z (pre-registered test type), Wald 95% CI.

    .venv/Scripts/python.exe scripts/posthoc/s07_spec_curve.py
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from _common import FIGDIR, mpl, save, two_prop
from _relabel import Prices
from impi.pipeline.p2_events import load_events
from impi.validation.splits import assign_split, split_bounds

BARRIERS = [(1.5, 1.0), (2.0, 1.0), (3.0, 1.5)]
HORIZONS = [5, 10, 20]
THRESH = [0.0, 0.5, 1.0]
SETS = {"BROAD": "broad", "B1": "all"}
FROZEN = {"barrier": "+2/-1", "horizon": 10, "threshold": 0.0, "event_set": "BROAD"}


def main():
    ev = load_events()
    a, b = split_bounds()["test"]
    pool = ev[ev["tier1"] & ev["eligible"] & ev["family"].isin(SETS.values()) & (ev["date"] >= a) & (ev["date"] <= b)].copy()
    P = Prices(set(pool["symbol"].astype(str)))
    rows = []
    for (up, lo), hz in itertools.product(BARRIERS, HORIZONS):
        lab = P.label(pool, up, lo, hz)
        d_all = pool.join(lab)
        d_all = d_all[(assign_split(d_all["date"], d_all["new_label_end"]) == "test") & d_all["label"].notna()]
        for sname, fam in SETS.items():
            d = d_all[d_all["family"] == fam]
            spec = {"barrier": f"+{up:g}/-{lo:g}", "horizon": hz, "event_set": sname, "base_rate": float(d["label"].mean())}
            h1 = d[d["deliv_z20_prev"].notna()]
            for th in THRESH:
                e = h1["deliv_z20_prev"] > th
                r = two_prop(int(h1.loc[e, "label"].sum()), int(e.sum()), int(h1.loc[~e, "label"].sum()), int((~e).sum()))
                rows.append({"hypothesis": "H1", **spec, "threshold": th, **r, "expected_sign": "+"})
            h2 = d[d["ret_20"].notna() & d["oi_chg20"].notna()]
            e = (h2["ret_20"] > 0) & (h2["oi_chg20"] < 0)
            r = two_prop(int(h2.loc[e, "label"].sum()), int(e.sum()), int(h2.loc[~e, "label"].sum()), int((~e).sum()))
            rows.append({"hypothesis": "H2", **spec, "threshold": None, **r, "expected_sign": "-"})
    df = pd.DataFrame(rows)
    df["significant_5pct"] = df["p"] < 0.05
    df["sig_predicted_direction"] = df["significant_5pct"] & np.where(df["expected_sign"] == "+", df["diff"] > 0, df["diff"] < 0)
    df["sig_opposite_direction"] = df["significant_5pct"] & ~df["sig_predicted_direction"]
    df["is_frozen_spec"] = ((df["barrier"] == FROZEN["barrier"]) & (df["horizon"] == FROZEN["horizon"]) &
                            (df["event_set"] == FROZEN["event_set"]) & ((df["threshold"] == 0.0) | df["threshold"].isna()))
    from impi.validation.multiple_testing import benjamini_hochberg
    from _common import load
    d_star = load("B_breakeven.json")["break_even"]["break_even_success_rate_change_pp"]
    df["bh_q_within_hypothesis"] = np.nan
    for h, g in df.groupby("hypothesis"):
        q, _ = benjamini_hochberg(g["p"].to_numpy(), 0.10)
        df.loc[g.index, "bh_q_within_hypothesis"] = q
    df["rank_by_effect_desc"] = df.groupby("hypothesis")["diff"].rank(ascending=False, method="min").astype(int)
    summary = {}
    for h, g in df.groupby("hypothesis"):
        sgn = 1 if h == "H1" else -1                      # predicted direction
        eff = sgn * 100 * g["diff"]
        fz = g[g["is_frozen_spec"]].iloc[0]
        summary_extra = {
            "n_bh_q10_within_hypothesis": int((g["bh_q_within_hypothesis"] <= 0.10).sum()),
            "n_bh_q05_within_hypothesis": int((g["bh_q_within_hypothesis"] <= 0.05).sum()),
            "share_predicted_direction": float((eff > 0).mean()),
            "max_effect_predicted_dir_pp": float(eff.max()),
            "median_effect_predicted_dir_pp": float(eff.median()),
            "break_even_pp": d_star,
            "n_upper_ci95_above_break_even": int(((sgn * 100 * np.array([c[1] if sgn > 0 else c[0] for c in g["ci95"]])) > d_star).sum()),
            "frozen_rank_in_predicted_direction": int((eff > float(sgn * 100 * fz["diff"])).sum() + 1),
            "frozen_percentile_in_predicted_direction": float(100 * (eff < float(sgn * 100 * fz["diff"])).mean()),
        }
        summary[h] = summary_extra
    for h, g in df.groupby("hypothesis"):
        summary[h] = {**summary[h], "n_specs": int(len(g)), "share_significant_5pct": float(g["significant_5pct"].mean()),
                      "n_significant": int(g["significant_5pct"].sum()),
                      "n_sig_predicted_direction": int(g["sig_predicted_direction"].sum()),
                      "n_sig_opposite_direction": int(g["sig_opposite_direction"].sum()),
                      "median_effect_pp": float(100 * g["diff"].median()),
                      "min_effect_pp": float(100 * g["diff"].min()), "max_effect_pp": float(100 * g["diff"].max()),
                      "share_positive": float((g["diff"] > 0).mean()),
                      "frozen_spec_effect_pp": float(100 * g.loc[g["is_frozen_spec"], "diff"].iloc[0])}
    save({"task": "C specification curve", "grid": {"barriers": [f"+{u:g}/-{l:g}" for u, l in BARRIERS], "horizons": HORIZONS,
                                                     "h1_thresholds": THRESH, "event_sets": list(SETS)},
          "frozen_spec": FROZEN, "summary": summary, "rows": df.to_dict("records"),
          "note": "Specs share events and overlap heavily, so they are not independent tests; the share significant is "
                  "descriptive. H2 has no delivery threshold, hence 18 specs."}, "C_spec_curve.json")
    print(pd.DataFrame(summary).T.to_string())
    plot(df)


def plot(df):
    """Drawn at the printed width (7.16 in) with 8 pt labels; significance is shown by marker SHAPE as well as
    colour so the figure reads in greyscale."""
    plt = mpl()
    fig = plt.figure(figsize=(7.16, 3.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.2], width_ratios=[54, 18 + 6], hspace=0.10, wspace=0.42,
                          left=0.14, right=0.99, top=0.885, bottom=0.135)
    styles = {"pred": dict(marker="^", c="0.05", s=22), "opp": dict(marker="v", c="0.05", s=22),
              "ns": dict(marker="o", facecolors="white", edgecolors="0.35", s=14)}
    for col, h in enumerate(["H1", "H2"]):
        g = df[df["hypothesis"] == h].sort_values("diff").reset_index(drop=True)
        x = np.arange(len(g))
        ax = fig.add_subplot(gs[0, col])
        lo = 100 * np.array([c[0] for c in g["ci95"]])
        hi = 100 * np.array([c[1] for c in g["ci95"]])
        kind = np.where(g["sig_predicted_direction"], "pred", np.where(g["sig_opposite_direction"], "opp", "ns"))
        ax.vlines(x, lo, hi, colors=np.where(kind == "ns", "0.6", "0.1"), lw=0.8,
                  linestyles=[("solid" if k != "ns" else (0, (2, 1))) for k in kind])
        for k, st in styles.items():
            m = kind == k
            if m.any():
                ax.scatter(x[m], 100 * g.loc[m, "diff"], zorder=3, linewidths=0.8, **st)
        fz = g["is_frozen_spec"].to_numpy()
        ax.scatter(x[fz], 100 * g.loc[fz, "diff"], marker="s", s=60, facecolors="none", edgecolors="k", linewidths=1.2,
                   zorder=4)
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xticks([])
        ax.set_xlim(-1, len(g))
        med = f"{100 * g['diff'].median():+.1f}".replace("-", "−")
        ax.set_title(f"{h} (post hoc), {len(g)} specs\n"
                     f"{int(g['significant_5pct'].sum())} with p < .05; median {med} pp", fontsize=8)
        if col == 0:
            ax.set_ylabel("effect (pp), 95% CI")
        axi = fig.add_subplot(gs[1, col], sharex=ax)
        dims = [("barrier", ["+1.5/-1", "+2/-1", "+3/-1.5"]), ("horizon", [5, 10, 20]), ("event_set", ["BROAD", "B1"])]
        if h == "H1":
            dims.insert(2, ("threshold", [0.0, 0.5, 1.0]))
        labels, yy = [], 0
        for name, vals in dims:
            for v in vals:
                m = (g[name] == v).to_numpy()
                axi.scatter(x[m], np.full(m.sum(), yy), s=6, marker="s", color="0.15")
                lab = (f"barrier {v}" if name == "barrier" else f"h = {v} d" if name == "horizon"
                       else f"z > {v:g}" if name == "threshold" else f"{v}")
                labels.append(lab.replace("-", "−") + (" *" if v in (FROZEN["barrier"], FROZEN["horizon"], FROZEN["event_set"])
                                                          or (name == "threshold" and v == 0.0) else ""))
                yy -= 1
        axi.set_yticks(np.arange(0, yy, -1))
        axi.set_yticklabels(labels, fontsize=8)
        axi.set_xticks([])
        axi.spines["bottom"].set_visible(False)
        axi.set_ylim(yy + 0.5, 0.5)
    fig.text(0.01, 0.06, "▲ significant (p < .05) in the predicted direction;  ▼ significant against it;  "
                          "○ not significant (dashed CI);  □ pre-registered (frozen) specification.",
             fontsize=8)
    fig.text(0.01, 0.012, "* marks the frozen choice in each dimension. Specifications share events and are not "
                          "independent tests.", fontsize=8)
    fig.savefig(FIGDIR / "fig_spec_curve.pdf", bbox_inches=None)


if __name__ == "__main__":
    main()
