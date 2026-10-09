"""P7 - robustness and validation suite (spec section 12). Uses TRAIN + VALIDATION data; the test
period's numbers come only from the one P6 run (re-reported, never recomputed).

    python -m impi.pipeline.p7_robustness

  * sub-periods 2015-19 / 2021-22 (2023+ = the P6 result)        * FII vs DII tested separately
  * COVID sensitivity (exclude Mar-May 2020)                      * placebo: delivery % shuffled across
  * VCP parameter grid (full surface, every point logged)           stocks within each day (20 draws)
  * 104-week COT sensitivity for the composite                    * Tier-2 replication (outside F&O)
  * EXPLORATORY lead-lag: FII positioning extremes and Granger tests (supporting evidence only)
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from impi.common import METRICS, config, log, log_result, rng, save_json, seed
from impi.features.build import OUT
from impi.features.delivery import delivery_features
from impi.features.vcp import VCPDetector
from impi.labels.events import apply_cooldown
from impi.models.composite import Composite
from impi.pipeline.common_phase import assert_preregistered, tier1, tier2_only
from impi.pipeline.p2_events import assert_frozen, load_events
from impi.pipeline.p4_composite import MODELS
from impi.research.hypothesis_tests import amihud_buckets, run_family, test_h1, test_h3
from impi.transforms import pit_store as pit

TV = ["train", "validation"]


def _between(d, a, b):
    return d[(d["date"] >= pd.Timestamp(a)) & (d["date"] <= pd.Timestamp(b))]


def placebo(ev: pd.DataFrame, cuts, draws: int = 20) -> dict:
    fcfg = config()["features"]
    fams = {"STRICT": tier1(ev, "strict", TV), "BROAD": tier1(ev, "broad", TV)}
    syms = sorted(set(pd.concat(fams.values())["symbol"].astype(str)))
    eq = pit.read("equity_adj", columns=["date", "symbol", "series", "close", "deliv_pct"], symbols=syms)
    eq["symbol"] = eq["symbol"].astype(str)
    eq = eq.sort_values(["symbol", "date"]).reset_index(drop=True)
    actual = {k: test_h1(v, cuts, 200, rng(700))["diff"] for k, v in fams.items()}
    sims = {k: [] for k in fams}
    for i in range(draws):
        r = rng(710 + i)
        shuffled = eq.copy()
        m = shuffled["series"] == "EQ"
        shuffled.loc[m, "deliv_pct"] = shuffled.loc[m].groupby("date")["deliv_pct"].transform(
            lambda s: s.sample(frac=1.0, random_state=int(r.integers(1_000_000_000))).to_numpy())
        z = []
        for s, g in shuffled.groupby("symbol"):
            f = delivery_features(g.reset_index(drop=True), fcfg)
            z.append(pd.DataFrame({"symbol": s, "date": g["date"].to_numpy(), "dz_prev": f["deliv_z20_prev"].to_numpy()}))
        z = pd.concat(z)
        for k, v in fams.items():
            vv = v.drop(columns=["deliv_z20_prev"]).merge(z.rename(columns={"dz_prev": "deliv_z20_prev"}),
                                                            on=["symbol", "date"], how="left")
            sims[k].append(test_h1(vv, cuts, 50, rng(720 + i))["diff"])
    return {k: {"actual_diff": actual[k], "placebo_mean": float(np.nanmean(sims[k])),
                "placebo_sd": float(np.nanstd(sims[k])), "placebo_draws": sims[k],
                "share_placebo_ge_actual": float(np.mean(np.array(sims[k]) >= actual[k]))} for k in fams}


def vcp_grid(ev: pd.DataFrame, cuts) -> list[dict]:
    fcfg = config()["features"]
    base = pd.read_parquet(OUT / "base.parquet")
    from impi.validation.splits import assign_split
    base["split"] = assign_split(base["date"], base["label_end"])
    base = base[base["eligible"] & base["split"].isin(TV) & base["label_10"].notna()]
    keyset = base.set_index(["symbol", "date"])
    syms = sorted(base["symbol"].astype(str).unique())
    eq = pit.read("equity_adj", columns=["date", "symbol", "series", "high", "low", "close", "volume", "deliv_pct"], symbols=syms)
    eq["symbol"] = eq["symbol"].astype(str)
    data = {}
    for s, g in eq.sort_values(["symbol", "date"]).groupby("symbol"):
        g = g.reset_index(drop=True)
        v = g["volume"].astype(float)
        data[s] = (g, delivery_features(g, fcfg)["deliv_z20_prev"].to_numpy(), v / v.shift(1).rolling(20, min_periods=15).mean())
    rows = []
    cooldown = config()["labels"]["events"]["strict"]["cooldown_days"]
    for k in [2, 3, 5]:
        dets = {s: VCPDetector(g.high, g.low, g.close, g.volume, {**fcfg["vcp"], "fractal_k": k}) for s, (g, _, _) in data.items()}
        for depth in [0.4, 0.5, 0.6]:
            for vm in [1.25, 1.5, 2.0]:
                for bmin, bmax in [(20, 60), (25, 75), (30, 90)]:
                    cfg = {**fcfg["vcp"], "fractal_k": k, "max_depth": depth, "volume_mult": vm,
                           "base_min_days": bmin, "base_max_days": bmax}
                    recs = []
                    for s, (g, dz, vr) in data.items():
                        det = dets[s]
                        det.cfg = cfg
                        trig = apply_cooldown(np.array([t for t, _ in det.triggers(vr.to_numpy())], int), cooldown)
                        for t in trig:
                            if g.loc[t, "series"] == "EQ":
                                recs.append((s, g.loc[t, "date"], dz[t]))
                    d = pd.DataFrame(recs, columns=["symbol", "date", "deliv_z20_prev"])
                    d = d.join(keyset[["label_10"]], on=["symbol", "date"], how="inner")
                    name = f"k{k}_d{depth}_v{vm}_b{bmin}-{bmax}"
                    is_spec = (k, depth, vm, bmin, bmax) == (2, 0.5, 1.5, 25, 75)
                    row = {"config": name, "spec_default": is_spec, "n": len(d),
                           "success_rate": float(d["label_10"].mean()) if len(d) else np.nan}
                    if len(d) > 30:
                        e = d["deliv_z20_prev"] > 0
                        row["h1_diff"] = float(d.loc[e, "label_10"].mean() - d.loc[~e, "label_10"].mean())
                    rows.append(row)
                    log_result("P7", "vcp_grid", name, "train+validation", len(d), "success_rate", row["success_rate"],
                               row, seed_value=None)
    return rows


def lead_lag(ev_market: pd.DataFrame) -> dict:
    """EXPLORATORY (secondary): FII stock-futures positioning extremes vs forward Nifty 500 returns,
    and Granger tests on stationary transforms. Train+validation period only. Not causal evidence."""
    from statsmodels.tsa.stattools import grangercausalitytests
    m = pd.read_parquet(OUT / "market.parquet")
    idx = pit.read("index")
    n500 = idx[idx["index"] == "nifty500"].set_index("date")["close"].sort_index()
    a, b = pd.Timestamp(config()["splits"]["train"][0]), pd.Timestamp(config()["splits"]["validation"][1])
    m = m[(m["date"] >= a) & (m["date"] <= b)].set_index("date")
    out = {"label": "EXPLORATORY - secondary, not part of the pre-registered family"}
    for h in [5, 10, 20, 60]:
        fwd = (n500.shift(-h) / n500 - 1).reindex(m.index)
        for col in ["fii_fut_stk_cot", "fii_fut_idx_cot"]:
            hi, lo = m[col] > 90, m[col] < 10
            out[f"{col}_h{h}"] = {"uncond": float(fwd.mean()), "cot_gt_90": float(fwd[hi].mean()), "n_gt_90": int(hi.sum()),
                                  "cot_lt_10": float(fwd[lo].mean()), "n_lt_10": int(lo.sum())}
    r = np.log(n500).diff().reindex(m.index)
    for col in ["fii_fut_stk_cot", "dii_fut_stk_cot", "fii_fut_idx_cot"]:
        df = pd.concat([r.rename("ret"), m[col].diff().rename("dcot")], axis=1).dropna()
        res = {}
        for direction, cols in [("cot_to_ret", ["ret", "dcot"]), ("ret_to_cot", ["dcot", "ret"])]:
            g = grangercausalitytests(df[cols].to_numpy(), maxlag=5)
            res[direction] = {lag: float(g[lag][0]["ssr_ftest"][1]) for lag in g}
        out[f"granger_{col}"] = res
    return out


def main():
    assert_preregistered()
    assert_frozen()
    ev = load_events()
    cuts = amihud_buckets(tier1(ev, "broad", ["train"]))
    r = rng(800)
    out = {}
    sp = config()["splits"]["subperiods"]
    out["subperiods"] = {}
    for name, (a, b) in sp.items():
        if name == "2023+":
            p6 = METRICS / "p6_test.json"
            out["subperiods"][name] = ("see p6_test.json (the one test-set run)" if p6.exists() else "P6 not run")
            continue
        fams = {"STRICT": _between(tier1(ev, "strict", TV), a, b), "BROAD": _between(tier1(ev, "broad", TV), a, b)}
        out["subperiods"][name] = run_family(fams, cuts, r)
    out["fii_vs_dii_h3"] = {f"{fam}_{w}": test_h3(tier1(ev, fam.lower(), TV), 500, r, which=w)
                            for fam in ["STRICT", "BROAD"] for w in ["both", "fii_only", "dii_only"]}
    c0, c1 = config()["splits"]["covid_exclusion"]
    no_covid = ev[~((ev["date"] >= pd.Timestamp(c0)) & (ev["date"] <= pd.Timestamp(c1)))]
    out["covid_excluded"] = run_family({"STRICT": tier1(no_covid, "strict", TV), "BROAD": tier1(no_covid, "broad", TV)}, cuts, r)
    log("P7: placebo ...")
    out["placebo"] = placebo(ev, cuts)
    log("P7: VCP grid ...")
    out["vcp_grid"] = vcp_grid(ev, cuts)
    # 104-week COT sensitivity for the composite (validation AUC)
    va = tier1(ev, "broad", ["validation"])
    tr = tier1(ev, "broad", ["train"])
    with open(MODELS / "composite.pkl", "rb") as f:
        comp = pickle.load(f)
    alt_tr = tr.assign(fii_fut_stk_cot=tr["fii_fut_stk_cot_104w"])
    alt_va = va.assign(fii_fut_stk_cot=va["fii_fut_stk_cot_104w"])
    comp104 = Composite().fit(alt_tr)
    out["cot_104w_sensitivity"] = {
        "val_auc_52w": float(roc_auc_score(va["label_10"], comp.score(va)["score"])),
        "val_auc_104w": float(roc_auc_score(va["label_10"], comp104.score(alt_va)["score"]))}
    out["tier2_replication_trainval"] = {fam: test_h1(tier2_only(ev, fam, TV), cuts, 500, r)
                                         for fam in ["strict", "broad"] if len(tier2_only(ev, fam, TV)) > 30}
    out["lead_lag_exploratory"] = lead_lag(ev)
    save_json(out, METRICS / "p7_robustness.json")
    log("P7 done")


if __name__ == "__main__":
    main()
