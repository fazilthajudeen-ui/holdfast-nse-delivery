"""POST-HOC task G: why do test-set breakouts (B1, 24.9%) succeed less often than all Tier-1 stock-days
(B0, 26.7%)?  Compares B1 events with non-breakout stock-days on the T+1 opening gap, the ATR level and the
T+1 range, and measures how many losses are same-day barrier ties (both barriers touched on the resolving
bar; the frozen rule D-016 counts them as losses). Re-labelling uses the frozen triple_barrier(); its
agreement with the frozen label_10 is reported.

    .venv/Scripts/python.exe scripts/posthoc/s06_breakout_ties.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from _common import save, two_prop
from _relabel import Prices
from impi.common import METRICS
from impi.features.build import OUT
from impi.pipeline.common_phase import tier1
from impi.pipeline.p2_events import load_events
from impi.validation.splits import assign_split


def summ(d: pd.DataFrame) -> dict:
    lab = d["label"]
    losses = lab == 0
    return {"n": int(len(d)), "success_rate": float(lab.mean()),
            "frozen_success_rate": float(d["label_10"].mean()),
            "tie_share_of_all": float(d["tie"].mean()), "tie_share_of_losses": float(d.loc[losses, "tie"].mean()),
            "success_rate_ties_excluded": float(d.loc[~d["tie"], "label"].mean()),
            "success_rate_ties_as_half": float((lab + 0.5 * d["tie"]).mean()),
            "day1_stop_share": float(((d["exit_k"] == 1) & losses & ~d["tie"]).mean()),
            "mean_gap_pct": float(d["gap_pct"].mean()), "median_gap_pct": float(d["gap_pct"].median()),
            "mean_abs_gap_atr": float(d["gap_atr"].abs().mean()), "mean_gap_atr": float(d["gap_atr"].mean()),
            "mean_atr_pct": float(d["atr_pct"].mean()), "median_atr_pct": float(d["atr_pct"].median()),
            "mean_day1_range_atr": float(d["day1_range_atr"].mean())}


def main():
    p6 = json.loads((METRICS / "p6_test.json").read_text())
    ev = load_events()
    b1 = tier1(ev, "all", ["test"])
    base = pd.read_parquet(OUT / "base.parquet")
    base["split"] = assign_split(base["date"], base["label_end"])
    b0 = base[base["eligible"] & (base["split"] == "test") & base["label_10"].notna()].copy()
    key = lambda d: d["symbol"].astype(str) + "|" + pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    b0["is_breakout"] = key(b0).isin(set(key(b1)))
    P = Prices(set(b0["symbol"].astype(str)))
    lab0 = P.label(b0, 2.0, 1.0, 10)
    b0 = b0.join(lab0)
    agree = float((b0["label"] == b0["label_10"]).mean())
    nb = b0[~b0["is_breakout"]]
    bk = b0[b0["is_breakout"]]
    lab1 = P.label(b1, 2.0, 1.0, 10)
    b1 = b1.join(lab1)
    out = {"task": "G breakout vs base", "label_agreement_with_frozen_label_10_B0": agree,
           "label_agreement_with_frozen_label_10_B1": float((b1["label"] == b1["label_10"]).mean()),
           "locked_ladder": {r["rung"]: r["success_rate"] for r in p6["ladder_broad"]},
           "B0_all_stock_days": summ(b0), "B1_breakout_events": summ(b1),
           "non_breakout_days": summ(nb), "breakout_days_in_B0": summ(bk)}
    # decomposition: success-rate gap with ties counted as losses (frozen) vs ties excluded
    g_frozen = out["B1_breakout_events"]["success_rate"] - out["non_breakout_days"]["success_rate"]
    g_excl = out["B1_breakout_events"]["success_rate_ties_excluded"] - out["non_breakout_days"]["success_rate_ties_excluded"]
    g_half = out["B1_breakout_events"]["success_rate_ties_as_half"] - out["non_breakout_days"]["success_rate_ties_as_half"]
    out["gap_decomposition_pp"] = {"breakout_minus_nonbreakout_frozen_rule": 100 * g_frozen,
                                   "ties_excluded": 100 * g_excl, "ties_counted_as_half": 100 * g_half,
                                   "share_of_gap_explained_by_ties_excluded": (g_frozen - g_excl) / g_frozen if g_frozen else None}
    t = {}
    for v in ["gap_atr", "atr_pct", "day1_range_atr"]:
        a, b = b1[v].dropna(), nb[v].dropna()
        t[v] = {"breakout_mean": float(a.mean()), "nonbreakout_mean": float(b.mean()),
                "welch_t": float(stats.ttest_ind(a, b, equal_var=False).statistic),
                "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue)}
    tie_test = two_prop(int(b1["tie"].sum()), len(b1), int(nb["tie"].sum()), len(nb))
    out["tests_B1_vs_nonbreakout"] = {**t, "tie_rate": tie_test}
    out["caveat"] = ("Non-breakout days are overlapping stock-days (serially dependent), so the t-tests overstate precision; "
                     "they are descriptive.")
    save(out, "G_breakout_ties.json")
    print(json.dumps({k: v for k, v in out.items() if k not in ("caveat",)}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
