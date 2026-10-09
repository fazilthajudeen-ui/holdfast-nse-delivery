"""H1-H3 tests (spec section 7) exactly as pre-registered in pre_registered_hypotheses.md.

H1  exposure: deliv_z20_prev > 0 (delivery expansion during the base, trigger day excluded)
    outcome : label_10 (triple-barrier success).  expected: success(exposed) > success(rest)
H2  exposure: divergence = ret_20 > 0 and oi_chg20 < 0 (price up while stock futures OI falls)
    outcome : label_10.  expected: success(divergent) < success(rest)
H3  exposure: market regime FII-selling & DII-buying = fii_flow20 < 0 and dii_flow20 > 0
    outcome : return asymmetry = mean(MFE_10) / mean(|MAE_10|).  expected: higher in the regime
Primary test: two-proportion z (H1, H2) / month-block bootstrap of the asymmetry difference (H3).
Supporting: logistic regression with controls (size proxy, sector FE, VIX regime, Amihud liquidity
bucket) and the delivery x positioning interaction, with month-clustered standard errors.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from impi.common import config
from impi.research.stats import month_block_bootstrap, two_proportion_z


def _rate_diff(exposed_col: str):
    def f(d: pd.DataFrame) -> float:
        a, b = d.loc[d[exposed_col], "label_10"], d.loc[~d[exposed_col], "label_10"]
        return float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
    return f


def _asym(d: pd.DataFrame) -> float:
    mae = d["mae_10"].abs().mean()
    return float(d["mfe_10"].mean() / mae) if mae and mae > 0 else np.nan


def _asym_diff(col: str):
    def f(d: pd.DataFrame) -> float:
        a, b = d[d[col]], d[~d[col]]
        return _asym(a) - _asym(b) if len(a) and len(b) else np.nan
    return f


def amihud_buckets(train: pd.DataFrame) -> list[float]:
    return list(train["log_amihud20"].quantile([1 / 3, 2 / 3]).to_numpy())


def _controls(d: pd.DataFrame, cuts: list[float]) -> pd.DataFrame:
    d = d.copy()
    d["liq_bucket_ami"] = np.digitize(d["log_amihud20"].fillna(np.nanmedian(d["log_amihud20"])), cuts).astype(str)
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)
    d["sector"] = d["sector"].fillna("Unknown")
    return d


def _logit(formula: str, d: pd.DataFrame) -> dict:
    cols = [c for c in ["label_10", "log_turnover60", "vix_pct252"] if c in d]
    dd = d.dropna(subset=cols)
    for f in (formula, formula.replace(" + C(sector)", "")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.glm(f, data=dd, family=sm.families.Binomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.loc[:, "month"])[0]})
            keep = [k for k in m.params.index if not k.startswith("C(")]
            return {"formula": f, "n": int(m.nobs),
                    "coef": {k: float(m.params[k]) for k in keep},
                    "p": {k: float(m.pvalues[k]) for k in keep}}
        except Exception as e:                           # singular sector design -> retry without FE
            err = str(e)[:200]
    return {"formula": formula, "error": err}


def test_h1(ev: pd.DataFrame, cuts: list[float], n_boot: int, rng) -> dict:
    d = ev[ev["deliv_z20_prev"].notna() & ev["label_10"].notna()].copy()
    d["exposed"] = d["deliv_z20_prev"] > 0
    x1, n1 = int(d.loc[d.exposed, "label_10"].sum()), int(d.exposed.sum())
    x2, n2 = int(d.loc[~d.exposed, "label_10"].sum()), int((~d.exposed).sum())
    z = two_proportion_z(x1, n1, x2, n2)
    boot = month_block_bootstrap(d, _rate_diff("exposed"), n_boot, rng)
    d = _controls(d, cuts)
    d["deliv_exp"] = d["exposed"].astype(float)
    d["pos"] = (d["price_oi_align"] - d["price_oi_align"].mean()) / d["price_oi_align"].std()
    lg = _logit("label_10 ~ deliv_exp + pos + deliv_exp:pos + log_turnover60 + vix_pct252 + C(liq_bucket_ami) + C(sector)",
                d.dropna(subset=["pos"])) if d["pos"].notna().sum() > 50 else {"error": "no positioning data"}
    return {"hypothesis": "H1", "test": "two_proportion_z", **z, "expected_sign": "+",
            "direction_ok": bool(z["diff"] > 0) if np.isfinite(z["diff"]) else None,
            "bootstrap": boot, "logit": lg, "n": int(len(d))}


def test_h2(ev: pd.DataFrame, cuts: list[float], n_boot: int, rng) -> dict:
    d = ev[ev["ret_20"].notna() & ev["oi_chg20"].notna() & ev["label_10"].notna()].copy()
    d["exposed"] = (d["ret_20"] > 0) & (d["oi_chg20"] < 0)
    x1, n1 = int(d.loc[d.exposed, "label_10"].sum()), int(d.exposed.sum())
    x2, n2 = int(d.loc[~d.exposed, "label_10"].sum()), int((~d.exposed).sum())
    z = two_proportion_z(x1, n1, x2, n2)
    boot = month_block_bootstrap(d, _rate_diff("exposed"), n_boot, rng)
    d = _controls(d, cuts)
    d["div"] = d["exposed"].astype(float)
    d["dz"] = (d["deliv_z20"] - d["deliv_z20"].mean()) / d["deliv_z20"].std()
    lg = _logit("label_10 ~ div + dz + div:dz + log_turnover60 + vix_pct252 + C(liq_bucket_ami) + C(sector)",
                d.dropna(subset=["dz"])) if len(d) > 50 else {"error": "too few events"}
    return {"hypothesis": "H2", "test": "two_proportion_z", **z, "expected_sign": "-",
            "direction_ok": bool(z["diff"] < 0) if np.isfinite(z["diff"]) else None,
            "bootstrap": boot, "logit": lg, "n": int(len(d))}


def regime_flag(ev: pd.DataFrame, which: str = "both") -> pd.Series:
    fii_sell = ev["fii_flow20"] < 0
    dii_buy = ev["dii_flow20"] > 0
    return {"both": fii_sell & dii_buy, "fii_only": fii_sell, "dii_only": dii_buy}[which]


def test_h3(ev: pd.DataFrame, n_boot: int, rng, which: str = "both") -> dict:
    d = ev[ev["fii_flow20"].notna() & ev["dii_flow20"].notna() & ev["mfe_10"].notna() & ev["mae_10"].notna()].copy()
    d["exposed"] = regime_flag(d, which)
    boot = month_block_bootstrap(d, _asym_diff("exposed"), n_boot, rng)
    a, b = d[d.exposed], d[~d.exposed]
    z = two_proportion_z(int(a["label_10"].sum()), len(a), int(b["label_10"].sum()), len(b))
    return {"hypothesis": "H3" if which == "both" else f"H3_{which}", "test": "month_block_bootstrap_asymmetry",
            "asym_exposed": _asym(a) if len(a) else np.nan, "asym_rest": _asym(b) if len(b) else np.nan,
            "diff": boot["estimate"], "p": boot["p"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "n1": int(len(a)), "n2": int(len(b)), "expected_sign": "+",
            "direction_ok": bool(boot["estimate"] > 0) if np.isfinite(boot["estimate"]) else None,
            "success_rate_test": z, "mean_ret10_exposed": float(a["ret_tb_10"].mean()) if len(a) else np.nan,
            "mean_ret10_rest": float(b["ret_tb_10"].mean()) if len(b) else np.nan, "n": int(len(d))}


def run_family(events: dict[str, pd.DataFrame], cuts: list[float], rng) -> list[dict]:
    """events: {'STRICT': df, 'BROAD': df}. Returns the 6 pre-registered tests."""
    n_boot = int(config()["models"]["bootstrap"]["n"])
    out = []
    for fam, ev in events.items():
        for fn in (test_h1, test_h2):
            r = fn(ev, cuts, n_boot, rng)
            r["family"] = fam
            out.append(r)
        r = test_h3(ev, n_boot, rng)
        r["family"] = fam
        out.append(r)
    return out
