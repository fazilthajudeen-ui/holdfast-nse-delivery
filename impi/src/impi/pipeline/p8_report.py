"""P8 - figures and a results summary generated ONLY from the saved result files (no recomputation).

    python -m impi.pipeline.p8_report

Writes results/figures/*.png and results/summary.md. Every number in the summary is read from
results/metrics/*.json, so the write-up cannot drift from the computed results.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from impi.common import FIGURES, METRICS, RESULTS, log

# validated reference palette (dataviz skill): first three categorical slots + neutrals
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 150, "axes.axisbelow": True})


def _load(name):
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


def _pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{d}f}%"


def _num(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def fig_ladders(p6):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, key, title in [(axes[0], "ladder_strict", "STRICT (VCP) ladder"), (axes[1], "ladder_broad", "BROAD ladder")]:
        lad = p6[key]
        names = [r["rung"].replace("_", " ") for r in lad][::-1]
        raw = [r["success_rate"] for r in lad][::-1]
        vals = [v if v is not None and v == v else 0 for v in raw]
        ns = [r["n"] for r in lad][::-1]
        cols = [GREY if "B0" in n else BLUE for n in names]
        ax.barh(names, vals, color=cols, height=0.55)
        for i, (v, rv, n) in enumerate(zip(vals, raw, ns)):
            txt = "no events" if not n else f"{100 * v:.1f}%  (n={n:,})" + ("  [n<30]" if n < 30 else "")
            ax.text(v + 0.005, i, txt, va="center", fontsize=7.5, color=INK2)
        ax.set_title(title)
        ax.set_xlabel("Triple-barrier success rate, 10 days (test set)")
        ax.set_xlim(0, max(vals + [0.01]) * 1.75)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_benchmark_ladders_test.png")
    plt.close(fig)


def fig_h1(p3, p6):
    rows = []
    for src, data in [("Train+validation", p3["tests"]), ("Test", p6["confirmatory"])]:
        for t in data:
            if t["hypothesis"] == "H1":
                rows.append((f"{t['family']}\n{src}", t.get("p1"), t.get("p2"), t.get("n1"), t.get("n2")))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(rows))
    w = 0.36
    a = [r[1] if r[1] is not None and r[1] == r[1] else 0 for r in rows]
    b = [r[2] if r[2] is not None and r[2] == r[2] else 0 for r in rows]
    ax.bar(x - w / 2, a, w, color=BLUE, label="Delivery expanding during the base (z > 0)")
    ax.bar(x + w / 2, b, w, color=ORANGE, label="Not expanding")
    for i, r in enumerate(rows):
        for dx, val, n in [(-w / 2, a[i], r[3]), (w / 2, b[i], r[4])]:
            ax.text(i + dx, val + 0.005, f"n={n or 0:,}" if n else "no events", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("Success rate (10-day triple barrier)")
    ax.set_title("H1: breakout success by delivery behaviour during the base")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    ax.set_ylim(0, max(a + b + [0.01]) * 1.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_h1_delivery_success.png")
    plt.close(fig)


def fig_event_study(p3):
    st = p3["event_studies"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, fam in zip(axes, ["STRICT", "BROAD"]):
        rows = [r for r in st.get(f"{fam}_by_delivery", []) if r["n"] >= 30]
        if not rows:
            ax.text(0.5, 0.5, "insufficient data (n<30 per group)", ha="center", va="center", transform=ax.transAxes, color=INK2)
        for r, col in zip(sorted(rows, key=lambda z: z["group"]), [BLUE, ORANGE]):
            hs = [5, 10, 20, 60]
            m = [r.get(f"mean_ex_{h}") for h in hs]
            lo = [r.get(f"ci_low_{h}") for h in hs]
            hi = [r.get(f"ci_high_{h}") for h in hs]
            m = [v if v is not None else np.nan for v in m]
            err = [[mm - (l if l is not None else mm) for mm, l in zip(m, lo)], [(h if h is not None else mm) - mm for mm, h in zip(m, hi)]]
            ax.errorbar(hs, np.array(m) * 100, yerr=np.array(err) * 100, color=col, marker="o", ms=4, lw=2, capsize=3,
                        label=f"{r['group'].replace('_', ' ')} (n={r['n']:,})")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks([5, 10, 20, 60])
        ax.set_xlabel("Trading days after entry")
        ax.set_title(f"{fam} events, train+validation")
        ax.legend(fontsize=7.5)
        ax.set_ylabel("Mean excess return vs Nifty 500 (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_event_study_by_delivery.png")
    plt.close(fig)


def fig_reliability(p6):
    rel = pd.DataFrame(p6["reliability_test_xgb"])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="Perfect calibration")
    ax.plot(rel["mean_pred"], rel["frac_pos"], color=BLUE, marker="o", ms=5, lw=2, label="XGBoost (isotonic, test)")
    lim = max(rel["mean_pred"].max(), rel["frac_pos"].max()) * 1.15
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed success rate")
    ax.set_title("Calibration on the test set")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_reliability_test.png")
    plt.close(fig)


def fig_shap():
    p = METRICS / "p5_shap_importance.csv"
    if not p.exists():
        return
    imp = pd.read_csv(p).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cols = [ORANGE if f.startswith(("deliv", "dz")) else BLUE for f in imp["feature"]]
    ax.barh(imp["feature"], imp["mean_abs_shap"], color=cols, height=0.6)
    ax.set_xlabel("Mean |SHAP value| (validation sample)")
    ax.set_title("XGBoost: 15 most important features")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_shap_importance.png")
    plt.close(fig)


def fig_equity():
    p = METRICS / "variant_returns_test.parquet"
    if not p.exists():
        return
    r = pd.read_parquet(p)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    pick = [c for c in ["B2b_broad", "B3b_broad_composite", "B4b_broad_ml_top_decile"] if c in r]
    for c, col in zip(pick, [BLUE, ORANGE, AQUA]):
        ax.plot((1 + r[c].fillna(0)).cumprod(), color=col, lw=1.6, label=c.replace("_", " "))
    if "nifty500" in r:
        ax.plot((1 + r["nifty500"].fillna(0)).cumprod(), color=GREY, lw=1.4, label="Nifty 500 (price)")
    ax.set_ylabel("Growth of 1 (after costs)")
    ax.set_title("Test-period backtests (equal weight, max 20 positions, costs included)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_equity_curves_test.png")
    plt.close(fig)


def fig_vcp_grid(p7):
    g = pd.DataFrame(p7["vcp_grid"])
    if g.empty:
        return
    g["k_depth"] = g["config"].str.extract(r"(k\d_d[\d.]+)")
    g["vol_base"] = g["config"].str.extract(r"(v[\d.]+_b\d+-\d+)")
    piv = g.pivot_table(index="k_depth", columns="vol_base", values="n", aggfunc="first")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(piv.to_numpy(), cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > np.nanmax(piv.to_numpy()) * 0.6 else INK)
    fig.colorbar(im, ax=ax, label="STRICT events (train+validation)")
    ax.set_title("VCP parameter grid: STRICT event counts (train + validation)")
    ax.set_xlabel("v = volume multiple, b = base length (days)")
    ax.set_ylabel("k = fractal bars, d = max depth")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_vcp_grid_counts.png")
    plt.close(fig)


def fig_placebo(p7):
    pl = p7.get("placebo", {})
    if not pl:
        return
    fig, axes = plt.subplots(1, len(pl), figsize=(4 * len(pl), 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (fam, v) in zip(axes, pl.items()):
        draws = [d for d in v["placebo_draws"] if d is not None and d == d]
        ax.hist(draws, bins=12, color=GREY, edgecolor="white")
        ax.axvline(v["actual_diff"], color=ORANGE, lw=2, label=f"Actual: {100 * v['actual_diff']:.1f} pp")
        ax.set_title(f"Placebo (delivery shuffled), {fam}")
        ax.set_xlabel("H1 success-rate difference")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("Shuffles")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_placebo.png")
    plt.close(fig)


def summary(p2, p3, p4, p5, p6, p7) -> str:
    L = ["# IMPI — results summary", "",
         "All numbers below are read from `results/metrics/*.json` (generated by the pipeline).", ""]
    if p2:
        L += ["## Data and events (P2)", "", f"- Feature library v1.0: {p2['features_v1']} features (frozen).",
              f"- Tier-1 eligible events by family and split: `{json.dumps(p2['tier1_eligible_counts'])}`",
              f"- Class balance (Tier-1 eligible): `{json.dumps(p2['class_balance'])}`",
              f"- Power rule triggered (Tier-2 widening of H1): {p2['power_rule_triggered']}", ""]
    if p3:
        L += ["## Exploratory hypothesis tests — train + validation (P3)", "",
              "| Hypothesis | Family | n exposed | n rest | exposed | rest | diff | p | BH q | supported |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t in p3["tests"]:
            ex = t.get("p1", t.get("asym_exposed"))
            re = t.get("p2", t.get("asym_rest"))
            L.append(f"| {t['hypothesis']} | {t['family']} | {t.get('n1')} | {t.get('n2')} | {_num(ex)} | {_num(re)} | "
                     f"{_num(t['diff'])} | {_num(t['p'], 4)} | {_num(t.get('bh_q'), 4)} | {t['supported']} |")
        L.append("")
    if p5:
        L += ["## ML challenger (P5, validation)", "",
              f"- AUC: XGBoost {_num(p5['xgb_val']['auc'])}, Random Forest {_num(p5['rf_val']['auc'])}, logistic "
              f"{_num(p5['logit_val']['auc'])}, composite {_num(p5['composite_val_auc'])} (base rate {_pct(p5['base_rate_val'])}).",
              f"- SHAP novelty check: `{json.dumps(p5['shap_novelty_verdict'])}`", ""]
    if p6:
        L += ["## Confirmatory tests — TEST set, run once (P6)", "",
              "| Hypothesis | Family | n | diff | p | BH q | expected | supported |", "|---|---|---|---|---|---|---|---|"]
        for t in p6["confirmatory"]:
            L.append(f"| {t['hypothesis']} | {t['family']} | {t['n']} | {_num(t.get('diff'))} | {_num(t.get('p'), 4)} | "
                     f"{_num(t.get('bh_q'), 4)} | {t['expected_sign']} | {t['supported']} |")
        L += ["", "Excluding the Delivery Lens pilot window (D-011):", "",
              "| Hypothesis | Family | n | diff | p | BH q | supported |", "|---|---|---|---|---|---|---|"]
        for t in p6["confirmatory_without_pilot_window"]:
            L.append(f"| {t['hypothesis']} | {t['family']} | {t['n']} | {_num(t.get('diff'))} | {_num(t.get('p'), 4)} | "
                     f"{_num(t.get('bh_q'), 4)} | {t['supported']} |")
        m = p6["ml_test"]
        L += ["", f"- Test AUC: XGBoost {_num(m['xgb']['auc'])}, RF {_num(m['rf']['auc'])}, logistic {_num(m['logit']['auc'])}, "
                  f"composite {_num(m['composite_auc'])}; base rate {_pct(m['base_rate'])}; calibrated Brier (XGB) {_num(m['xgb']['brier_calibrated'], 4)}.",
              f"- Walk-forward (monthly refits, frozen hyper-parameters): `{json.dumps(p6['walk_forward'])}`",
              f"- SPA / White Reality Check across test variants vs Nifty 500: `{json.dumps(p6['spa'])}`", "",
              "| Backtest (test) | trades | CAGR | Sharpe | Sortino | max DD | hit rate | profit factor | DSR | note |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for b in p6["backtests"]:
            dsr = p6["deflated_sharpe"].get(b["variant"], {}).get("dsr")
            L.append(f"| {b['variant']} | {b.get('n_trades')} | {_pct(b.get('cagr'))} | {_num(b.get('sharpe'), 2)} | "
                     f"{_num(b.get('sortino'), 2)} | {_pct(b.get('max_dd'))} | {_pct(b.get('hit_rate'))} | "
                     f"{_num(b.get('profit_factor'), 2)} | {_num(dsr, 3)} | {b.get('label', '')} |")
        bm = p6.get("benchmark_nifty500_test", {})
        L += [f"| Nifty 500 (price) | - | {_pct(bm.get('cagr'))} | {_num(bm.get('sharpe'), 2)} | {_num(bm.get('sortino'), 2)} | "
              f"{_pct(bm.get('max_dd'))} | - | - | - | benchmark |", ""]
    if p7:
        L += ["## Robustness (P7, train + validation)", "",
              f"- Placebo (delivery shuffled within each day, 20 draws): `{json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'placebo_draws'} for k, v in p7['placebo'].items()})}`",
              f"- 104-week COT sensitivity (composite, validation AUC): `{json.dumps(p7['cot_104w_sensitivity'])}`",
              f"- VCP grid: {len(p7['vcp_grid'])} parameter points, all logged; see fig7.", ""]
    L += ["## Figures", ""] + [f"- `{p.name}`" for p in sorted(FIGURES.glob("*.png"))]
    return "\n".join(L) + "\n"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    p2, p3, p4, p5, p6, p7 = (_load(n) for n in ["p2_summary.json", "p3_hypotheses.json", "p4_composite.json",
                                                 "p5_ml.json", "p6_test.json", "p7_robustness.json"])
    if p6:
        fig_ladders(p6)
        fig_reliability(p6)
    if p3 and p6:
        fig_h1(p3, p6)
    if p3:
        fig_event_study(p3)
    fig_shap()
    fig_equity()
    if p7:
        fig_vcp_grid(p7)
        fig_placebo(p7)
    (RESULTS / "summary.md").write_text(summary(p2, p3, p4, p5, p6, p7), encoding="utf-8")
    log(f"P8: {len(list(FIGURES.glob('*.png')))} figures, summary.md written")


if __name__ == "__main__":
    main()
