"""POST-HOC task B: are the test-set success-rate differences economically meaningful?

Cost break-even (derived from config/costs.yaml via backtest/costs.py, the model used by every backtest):
a success-rate change d raises expected gross return per trade by d x (mean return of successes - mean
return of failures). The d whose gain just pays one round trip is  d* = round-trip cost / spread.

For H1, H2 (BROAD and STRICT; full test and without the pilot window), the composite rule (score >= 1.0)
and the composite / ML top deciles, report 90% and 95% CIs (Wald and calendar-month block bootstrap) of
the exposed-minus-rest success-rate difference, and whether the largest effect in the PREDICTED direction
still inside the CI is below d*.

    .venv/Scripts/python.exe scripts/posthoc/s02_breakeven.py
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from _common import save, two_prop
from impi.backtest import costs as C
from impi.common import METRICS, config
from impi.models import ml
from impi.pipeline.common_phase import tier1
from impi.pipeline.p2_events import load_events
from impi.pipeline.p4_composite import MODELS

N_BOOT = 2000
SEED = 20261112


def block_ci(d: pd.DataFrame, col: str, seed: int) -> dict:
    r = np.random.default_rng(seed)
    groups = [g for _, g in d.groupby(d["date"].dt.to_period("M"))]
    vals = []
    for _ in range(N_BOOT):
        s = pd.concat([groups[i] for i in r.integers(0, len(groups), len(groups))])
        e = s[col]
        if e.any() and (~e).any():
            vals.append(s.loc[e, "label_10"].mean() - s.loc[~e, "label_10"].mean())
    v = np.array(vals)
    return {"ci90": [float(np.percentile(v, 5)), float(np.percentile(v, 95))],
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "n_boot": int(len(v))}


def main():
    p6 = json.loads((METRICS / "p6_test.json").read_text())
    blob = pickle.loads((MODELS / "ml_frozen.pkl").read_bytes())
    with open(MODELS / "composite.pkl", "rb") as f:
        comp = pickle.load(f)
    ev = load_events()
    pilot = pd.Timestamp(config()["splits"]["pilot_window_start"])

    # ---- break-even derivation
    cc = config()["costs"]
    notional = cc["initial_capital"] / cc["max_positions"]
    legs = {}
    for b in ["large", "mid", "small"]:
        legs[b] = {"buy_cost_pct": 100 * C.buy_cost(notional) / notional, "sell_cost_pct": 100 * C.sell_cost(notional) / notional,
                   "slippage_both_legs_pct": 2 * C.slippage_bps(b) / 100, "round_trip_pct": 100 * C.round_trip_pct(notional, b)}
    bt = next(b for b in p6["backtests"] if b["variant"] == "B2b_broad")
    mix = {k: v["count"] for k, v in bt["by_bucket"].items()}
    rt = sum(legs[k]["round_trip_pct"] * mix[k] for k in mix) / sum(mix.values())
    br = tier1(ev, "broad", ["test"])
    win, loss = br.loc[br.label_10 == 1, "ret_tb_10"], br.loc[br.label_10 == 0, "ret_tb_10"]
    spread = 100 * (win.mean() - loss.mean())
    d_star = rt / spread * 100                      # in percentage points of success rate
    derivation = {"notional_per_trade_rs": notional, "cost_components_by_bucket": legs, "bucket_mix_test_B2b_trades": mix,
                  "round_trip_cost_pct_weighted": rt,
                  "avg_success_return_pct": 100 * win.mean(), "avg_failure_return_pct": 100 * loss.mean(),
                  "n_success": int(len(win)), "n_failure": int(len(loss)), "spread_pp": spread,
                  "break_even_success_rate_change_pp": d_star,
                  "formula": "d* = round-trip cost / (avg success return - avg failure return)",
                  "backtest_trade_level_reference_B2b": {k: bt[k] for k in ["avg_win", "avg_loss", "hit_rate", "avg_net_ret", "n_trades"]},
                  "gross_vs_net_note": "Triple-barrier returns are gross (before costs) and measured from the T+1 open; the "
                                       "backtest avg_win/avg_loss are net of costs and slippage."}

    # ---- exposures
    br["ml_p"] = ml.predict("xgb", blob["models"]["xgb"], br[ml.FEATURES_V1].astype(float), blob["medians"])
    br["comp"] = comp.score(br)["score"]
    st = tier1(ev, "strict", ["test"])
    st["comp"] = comp.score(st)["score"]
    specs = []

    def add(name, d, col, sign, family, setname):
        d = d[d[col].notna()] if d[col].dtype != bool else d
        e = d[col].astype(bool)
        x1, n1 = int(d.loc[e, "label_10"].sum()), int(e.sum())
        x2, n2 = int(d.loc[~e, "label_10"].sum()), int((~e).sum())
        if min(n1, n2) == 0:
            specs.append({"test": name, "family": family, "set": setname, "expected_sign": sign, "n1": n1, "n2": n2,
                          "note": "not estimable: one group is empty"})
            return
        r = two_prop(x1, n1, x2, n2)
        bb = block_ci(d.assign(_e=e), "_e", SEED + len(specs))
        row = {"test": name, "family": family, "set": setname, "expected_sign": sign, **r,
               "ci90_block": bb["ci90"], "ci95_block": bb["ci95"], "small_sample": min(n1, n2) < 30}
        for lvl in ["90", "95"]:
            w, b = r[f"ci{lvl}"], bb[f"ci{lvl}"]
            # largest effect in the predicted direction still inside the CI (pp)
            row[f"max_effect_predicted_dir_pp_ci{lvl}_wald"] = 100 * (w[1] if sign == "+" else -w[0])
            row[f"max_effect_predicted_dir_pp_ci{lvl}_block"] = 100 * (b[1] if sign == "+" else -b[0])
            row[f"below_break_even_ci{lvl}"] = bool(max(row[f"max_effect_predicted_dir_pp_ci{lvl}_wald"],
                                                        row[f"max_effect_predicted_dir_pp_ci{lvl}_block"]) < d_star)
        specs.append(row)

    for setname, cond in [("full", lambda d: d), ("without_pilot", lambda d: d[d["date"] < pilot])]:
        for fam, d0 in [("BROAD", br), ("STRICT", st)]:
            d = cond(d0)
            h1 = d[d["deliv_z20_prev"].notna()].assign(_x=lambda x: x["deliv_z20_prev"] > 0)
            add("H1 delivery expansion", h1, "_x", "+", fam, setname)
            h2 = d[d["ret_20"].notna() & d["oi_chg20"].notna()].assign(_x=lambda x: (x["ret_20"] > 0) & (x["oi_chg20"] < 0))
            add("H2 price up, OI down", h2, "_x", "-", fam, setname)
            add("Composite rule (score >= 1.0)", d.assign(_x=d["comp"] >= comp.threshold), "_x", "+", fam, setname)
            add("Composite top decile (H4 cut)", d.assign(_x=d["comp"] >= blob["thresholds"]["composite_top_decile"]), "_x", "+", fam, setname)
            if fam == "BROAD":
                add("ML top decile (XGBoost)", d.assign(_x=d["ml_p"] >= blob["thresholds"]["ml_top_decile_raw"]), "_x", "+", fam, setname)
    out = {"task": "B break-even check", "break_even": derivation, "rows": specs,
           "reading": "below_break_even = even the most favourable end of the CI (Wald and block, whichever is larger) "
                      "is smaller than the success-rate change needed to pay one round trip."}
    save(out, "B_breakeven.json")
    df = pd.DataFrame(specs)
    print(f"break-even d* = {d_star:.2f} pp (RT {rt:.3f}% / spread {spread:.2f} pp)")
    print(df[["test", "family", "set", "n1", "n2", "diff", "ci95", "ci95_block", "below_break_even_ci95"]].to_string())


if __name__ == "__main__":
    main()
