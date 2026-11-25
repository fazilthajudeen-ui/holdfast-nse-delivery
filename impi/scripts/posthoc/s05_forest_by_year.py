"""POST-HOC task F: H1 effect by calendar year 2015-2026 (BROAD and STRICT Tier-1 eligible events, frozen
labels and exposure), labelled train / validation / test. 95% CIs: Wald and calendar-month block bootstrap.

    .venv/Scripts/python.exe scripts/posthoc/s05_forest_by_year.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import FIGDIR, mpl, save, two_prop
from impi.pipeline.common_phase import tier1
from impi.pipeline.p2_events import load_events

N_BOOT = 1000


def block_ci(d, seed):
    r = np.random.default_rng(seed)
    gs = [g for _, g in d.groupby(d["date"].dt.to_period("M"))]
    v = []
    for _ in range(N_BOOT):
        s = pd.concat([gs[i] for i in r.integers(0, len(gs), len(gs))])
        if s.e.any() and (~s.e).any():
            v.append(s.loc[s.e, "label_10"].mean() - s.loc[~s.e, "label_10"].mean())
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]


def main():
    ev = load_events()
    rows = []
    for fam in ["broad", "strict"]:
        d = tier1(ev, fam, ["train", "validation", "test"])
        d = d[d["deliv_z20_prev"].notna()].assign(e=lambda x: x["deliv_z20_prev"] > 0)
        groups = [(str(y), g) for y, g in d.groupby(d["date"].dt.year)]
        groups += [(f"pooled {sp}", d[d["split"] == sp]) for sp in ["train", "validation", "test"]]
        for k, (lab, g) in enumerate(groups):
            x1, n1 = int(g.loc[g.e, "label_10"].sum()), int(g.e.sum())
            x2, n2 = int(g.loc[~g.e, "label_10"].sum()), int((~g.e).sum())
            row = {"family": fam.upper(), "period": lab, "split": g["split"].iloc[0] if g["split"].nunique() == 1 else "mixed",
                   "n": len(g), "small_sample": min(n1, n2) < 30}
            if min(n1, n2) > 0:
                row.update(two_prop(x1, n1, x2, n2))
                row["ci95_block"] = block_ci(g, 700 + 13 * k + (0 if fam == "broad" else 500))
            rows.append(row)
    save({"task": "F H1 by year", "rows": rows,
          "note": "2015 starts in June (train start); 2020 includes the COVID months; 2026 runs to 9 Oct. Year rows are "
                  "descriptive; the pooled test row is the locked confirmatory H1 (diff -0.04 pp)."}, "F_h1_by_year.json")
    df = pd.DataFrame(rows)
    print(df[["family", "period", "split", "n", "n1", "n2", "diff", "ci95", "p"]].round(4).to_string())

    plt = mpl()
    col = {"train": "#4C72B0", "validation": "#DD8452", "test": "#C44E52", "mixed": "0.4"}
    b = df[df.family == "BROAD"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    y = np.arange(len(b))[::-1]
    for yi, (_, r) in zip(y, b.iterrows()):
        lo, hi = 100 * np.array(r["ci95"])
        pooled = r["period"].startswith("pooled")
        ax.errorbar(100 * r["diff"], yi, xerr=[[100 * r["diff"] - lo], [hi - 100 * r["diff"]]], fmt="D" if pooled else "o",
                    ms=4 if pooled else 3, color=col[r["split"]], elinewidth=0.9, capsize=1.5)
        ax.text(1.02, yi, f"{r['n1']}/{r['n2']}", transform=ax.get_yaxis_transform(), fontsize=6, va="center", color="0.3")
    ax.axvline(0, color="0.3", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(b["period"])
    ax.set_xlabel("H1: success rate, delivery-expanded − rest (pp), 95% CI")
    for sp in ["train", "validation", "test"]:
        ax.plot([], [], "o", color=col[sp], label=sp)
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    ax.set_title("H1 by year, BROAD events  [POST-HOC]\n(right: n exposed / n rest)")
    fig.savefig(FIGDIR / "fig_h1_forest.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
