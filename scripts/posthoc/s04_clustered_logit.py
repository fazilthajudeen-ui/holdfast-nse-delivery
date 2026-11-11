"""POST-HOC task E: eq. (3) on the TEST set with cluster-robust standard errors.

Exactly the frozen specification of research/hypothesis_tests.py (same controls, same standardisation):
  H1: label_10 ~ deliv_exp + pos + deliv_exp:pos + log_turnover60 + vix_pct252 + C(liq_bucket_ami) + C(sector)
  H2: label_10 ~ div + dz + div:dz + ...same controls
  beta1 = main exposure (D), beta2 = second factor (P), beta3 = interaction (D x P).
Standard errors (all reported): month-clustered (the pre-registered choice; reproduces p6_test.json),
date-clustered, symbol-clustered, and two-way date x symbol (Cameron, Gelbach & Miller 2011):
V = V_date + V_symbol - V_date&symbol.

    .venv/Scripts/python.exe scripts/posthoc/s04_clustered_logit.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod import families as sm_families

from _common import save
from impi.common import METRICS, config
from impi.pipeline.common_phase import tier1
from impi.pipeline.p2_events import load_events
from impi.research.hypothesis_tests import _controls, amihud_buckets

CTRL = " + log_turnover60 + vix_pct252 + C(liq_bucket_ami) + C(sector)"


def prep(d: pd.DataFrame, h: str, cuts) -> tuple[pd.DataFrame, str, list[str]]:
    if h == "H1":
        d = d[d["deliv_z20_prev"].notna() & d["label_10"].notna()].copy()
        d["exposed"] = d["deliv_z20_prev"] > 0
        d = _controls(d, cuts)
        d["deliv_exp"] = d["exposed"].astype(float)
        d["pos"] = (d["price_oi_align"] - d["price_oi_align"].mean()) / d["price_oi_align"].std()
        d = d.dropna(subset=["pos"])
        return d, "label_10 ~ deliv_exp + pos + deliv_exp:pos" + CTRL, ["deliv_exp", "pos", "deliv_exp:pos"]
    d = d[d["ret_20"].notna() & d["oi_chg20"].notna() & d["label_10"].notna()].copy()
    d["exposed"] = (d["ret_20"] > 0) & (d["oi_chg20"] < 0)
    d = _controls(d, cuts)
    d["div"] = d["exposed"].astype(float)
    d["dz"] = (d["deliv_z20"] - d["deliv_z20"].mean()) / d["deliv_z20"].std()
    d = d.dropna(subset=["dz"])
    return d, "label_10 ~ div + dz + div:dz" + CTRL, ["div", "dz", "div:dz"]


def fit(formula, d, groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.glm(formula, data=d, family=sm_families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})


def main():
    ev = load_events()
    cuts = amihud_buckets(tier1(ev, "broad", ["train"]))
    p6 = json.loads((METRICS / "p6_test.json").read_text())
    locked = {t["hypothesis"]: t.get("logit") for t in p6["confirmatory"] if t["family"] == "BROAD"}
    rows, notes = [], []
    pilot = pd.Timestamp(config()["splits"]["pilot_window_start"])
    for fam in ["broad", "strict"]:
        te = tier1(ev, fam, ["test"])
        for setname, d0 in [("full", te), ("without_pilot", te[te["date"] < pilot])]:
            for h in ["H1", "H2"]:
                d, formula, keys = prep(d0, h, cuts)
                d = d.dropna(subset=["log_turnover60", "vix_pct252"]).reset_index(drop=True)
                gid = {"month": pd.factorize(d["month"])[0], "date": pd.factorize(d["date"])[0],
                       "symbol": pd.factorize(d["symbol"].astype(str))[0]}
                gid["date_x_symbol"] = pd.factorize(d["date"].astype(str) + "|" + d["symbol"].astype(str))[0]
                used = formula
                try:
                    fits = {k: fit(formula, d, v) for k, v in gid.items()}
                except Exception as e:                       # singular sector design -> drop sector FE (same as frozen code)
                    used = formula.replace(" + C(sector)", "")
                    try:
                        fits = {k: fit(used, d, v) for k, v in gid.items()}
                    except Exception as e2:
                        notes.append({"family": fam.upper(), "set": setname, "hypothesis": h, "n": len(d),
                                      "error": f"not estimable: {str(e2)[:160]}"})
                        continue
                params = fits["month"].params
                ses_ok = np.isfinite(fits["month"].bse[keys]).all()
                separated = bool((params[keys].abs() > 10).any() or not ses_ok or len(d) < 100)
                status = ("NOT INTERPRETABLE: quasi/complete separation or n<100 (coefficients diverge; "
                          "cluster SEs unreliable with this few events)") if separated else "ok"
                V = {k: f.cov_params() for k, f in fits.items()}
                V["two_way_date_symbol"] = V["date"] + V["symbol"] - V["date_x_symbol"]
                for se_type in ["month", "date", "symbol", "two_way_date_symbol"]:
                    for j, k in enumerate(keys):
                        var = float(V[se_type].loc[k, k])
                        se = np.sqrt(var) if var > 0 else np.nan
                        b = float(params[k])
                        rows.append({"family": fam.upper(), "set": setname, "hypothesis": h, "formula": used,
                                     "n": int(fits["month"].nobs), "n_dates": int(d["date"].nunique()),
                                     "n_symbols": int(d["symbol"].nunique()), "se_type": se_type,
                                     "term": k, "beta": f"beta{j + 1}", "coef": b, "se": se,
                                     "ci95": [b - 1.959964 * se, b + 1.959964 * se], "p": float(2 * stats.norm.sf(abs(b / se))) if se == se else None,
                                     "odds_ratio": float(np.exp(b)),
                                     "locked_p6_coef": (locked.get(h) or {}).get("coef", {}).get(k) if fam == "broad" and setname == "full" else None,
                                     "locked_p6_p_month": (locked.get(h) or {}).get("p", {}).get(k) if fam == "broad" and setname == "full" else None,
                                     "small_sample": len(d) < 100, "status": status})
    save({"task": "E cluster-robust eq.(3) on the test set", "rows": rows, "not_estimable": notes,
          "note": "beta1/beta2/beta3 = main exposure, second factor, interaction. Two-way variance = V_date + V_symbol - "
                  "V_(date x symbol); each component uses statsmodels' cluster sandwich with small-sample correction."},
         "E_clustered_logit.json")
    df = pd.DataFrame(rows)
    print(df[(df.set == "full")][["family", "hypothesis", "se_type", "beta", "term", "n", "coef", "se", "p", "locked_p6_p_month"]].round(4).to_string())
    print(notes)


if __name__ == "__main__":
    main()
