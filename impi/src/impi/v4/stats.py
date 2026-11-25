"""v4 statistics exactly as fixed in pre_registered_v4.md (shared by the 2023-26 check and the live record)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from impi.research.stats import month_block_bootstrap, two_proportion_z
from impi.validation.multiple_testing import benjamini_hochberg

SEED = 20261012
N_BOOT = 2000
MIN_STOCKS = 30
MIN_GROUP = 30
EXPECTED = {"D1": "+", "D2": "+", "H2p": "-", "C1": "+", "V1": "+"}


def _two_sided(draws: np.ndarray) -> float:
    draws = draws[np.isfinite(draws)]
    if len(draws) < 10:
        return np.nan
    k = min((draws <= 0).sum(), (draws >= 0).sum())
    return float(min(1.0, 2 * (k + 1) / (len(draws) + 1)))


def circular_block_mean(x: np.ndarray, block: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Bootstrap distribution of the mean of a series, circular blocks of length `block`."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        out[b] = x[idx].mean()
    return out


def daily_ic(d: pd.DataFrame, signal: str, outcome: str) -> pd.Series:
    d = d[["date", signal, outcome]].dropna()
    g = d.groupby("date")
    ok = g.size() >= MIN_STOCKS
    ic = g.apply(lambda x: stats.spearmanr(x[signal], x[outcome])[0], include_groups=False)
    return ic[ok].dropna().sort_index()


def ic_test(d: pd.DataFrame, key: str, signal: str, outcome: str) -> dict:
    ic = daily_ic(d, signal, outcome)
    if len(ic) < 20:
        return {"hypothesis": key, "n_dates": int(len(ic)), "estimate": np.nan, "p": np.nan, "label": "insufficient data"}
    draws = circular_block_mean(ic.to_numpy(), 20)
    p = _two_sided(draws)                                      # share of bootstrap means on the other side of 0
    return {"hypothesis": key, "test": "mean daily Spearman IC, circular block bootstrap (20 days)",
            "signal": signal, "outcome": outcome, "estimate": float(ic.mean()), "n_dates": int(len(ic)),
            "n_obs": int(d[[signal, outcome]].dropna().shape[0]), "ic_t": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))),
            "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "share_dates_positive": float((ic > 0).mean()), "p": p, "label": "ok"}


def rate_test(ev: pd.DataFrame, key: str, exposed: str, outcome: str) -> dict:
    ev = ev[ev[outcome].notna()]
    a, b = ev[ev[exposed]], ev[~ev[exposed]]
    z = two_proportion_z(int(a[outcome].sum()), len(a), int(b[outcome].sum()), len(b))
    r = {"hypothesis": key, "test": "two-proportion z (two-sided); supporting month-block bootstrap",
         "outcome": outcome, "estimate": z["diff"], "rate_exposed": z["p1"], "rate_rest": z["p2"],
         "n1": len(a), "n2": len(b), "p": z["p"], "label": "ok"}
    if min(len(a), len(b)) < MIN_GROUP:
        r.update({"p_unguarded": r["p"], "p": np.nan, "label": f"insufficient data (n<{MIN_GROUP} in a group)"})
        return r
    bs = month_block_bootstrap(ev.assign(_e=ev[exposed]),
                               lambda x: float(x.loc[x["_e"], outcome].mean() - x.loc[~x["_e"], outcome].mean()),
                               N_BOOT, np.random.default_rng(SEED))
    r["bootstrap"] = bs
    return r


def c1_test(m: pd.DataFrame) -> dict:
    m = m[m["n500_fwd60"].notna() & m["fii_fut_stk_cot"].notna()].sort_values("date").reset_index(drop=True)
    on = m["c1_on"].to_numpy(bool)
    y = m["n500_fwd60"].to_numpy(float)
    r = {"hypothesis": "C1", "test": "mean(signal days) - mean(all days), circular block bootstrap (60 days)",
         "n_days": int(len(m)), "n_signal_days": int(on.sum())}
    if on.sum() < MIN_GROUP:
        return {**r, "estimate": np.nan, "p": np.nan, "label": f"insufficient data (<{MIN_GROUP} signal days)"}
    est = float(y[on].mean() - y.mean())
    rng = np.random.default_rng(SEED)
    n, block = len(y), 60
    nb = int(np.ceil(n / block))
    draws = []
    for _ in range(N_BOOT):
        idx = ((rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        o, yy = on[idx], y[idx]
        if o.sum() > 0:
            draws.append(yy[o].mean() - yy.mean())
    draws = np.array(draws)
    return {**r, "estimate": est, "mean_signal_days": float(y[on].mean()), "mean_all_days": float(y.mean()),
            "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "p": _two_sided(draws), "label": "ok"}


def family(results: list[dict]) -> list[dict]:
    keys = [i for i, r in enumerate(results) if r.get("p") is not None and np.isfinite(r.get("p", np.nan))]
    q, _ = benjamini_hochberg(np.array([results[i]["p"] for i in keys]), 0.10) if keys else (np.array([]), None)
    for r in results:
        r["expected_sign"] = EXPECTED[r["hypothesis"]]
        r["bh_q"] = np.nan
        r["direction_ok"] = None
        r["supported"] = False
    for i, qq in zip(keys, q):
        r = results[i]
        r["bh_q"] = float(qq)
        est = r.get("estimate", np.nan)
        r["direction_ok"] = bool(est > 0) if r["expected_sign"] == "+" else bool(est < 0)
        r["supported"] = bool(r["direction_ok"] and qq <= 0.10)
    return results


def run_all(sig: pd.DataFrame, mk: pd.DataFrame) -> list[dict]:
    """The five v4 tests on already-filtered data (eligible stock-days / market days in the window)."""
    res = [ic_test(sig, "D1", "deliv_z20", "move20"),
           ic_test(sig, "D2", "sds", "fwd_20"),
           rate_test(sig[sig["b1_event"]], "H2p", "h2_exposed", "label_20"),
           c1_test(mk),
           rate_test(sig[sig["v1_event"]], "V1", "v1_exposed", "label_10")]
    return family(res)
