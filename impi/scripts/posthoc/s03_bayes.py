"""POST-HOC task D: Bayes factors BF01 (evidence for "no difference") for H1 and H2 on the test set.

Model (Kass & Raftery 1995 two-binomial comparison):
  H0: p_exposed = p_rest = p,  p ~ Beta(a, b)
  H1: p_exposed, p_rest independent ~ Beta(a, b)
  BF01 = B(a+x1+x2, b+n1+n2-x1-x2) B(a,b) / [ B(a+x1, b+n1-x1) B(a+x2, b+n2-x2) ]   (binomial terms cancel)
Priors (all reported):
  default   Beta(1, 1)                       uniform; implied prior SD of the difference ~ 41 pp
  wide      Beta(0.5, 0.5)                   Jeffreys; heavier tails, SD of difference ~ 50 pp
  skeptical Beta(k*pi, k*(1-pi)), k = 100,   pi = train+validation success rate of the family;
            prior SD of the difference ~ 6 pp, i.e. the alternative predicts realistic small effects.
Cross-check: logit-difference model (Gronau, Raj & Wagenmakers 2021 'abtest' parameterisation)
  log OR ~ N(0, s^2) with s in {0.25 skeptical, 0.5, 1.0 default}, nuisance grand mean ~ N(logit(pi), 1),
  integrated numerically. Interpretation scale: Lee & Wagenmakers (2013).

    .venv/Scripts/python.exe scripts/posthoc/s03_bayes.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

from _common import save
from impi.common import config
from impi.pipeline.common_phase import tier1
from impi.pipeline.p2_events import load_events


def lbeta(a, b):
    return special.betaln(a, b)


def bf01_beta(x1, n1, x2, n2, a, b):
    l0 = lbeta(a + x1 + x2, b + n1 + n2 - x1 - x2) - lbeta(a, b)
    l1 = lbeta(a + x1, b + n1 - x1) - lbeta(a, b) + lbeta(a + x2, b + n2 - x2) - lbeta(a, b)
    return float(np.exp(l0 - l1))


def bf01_logit(x1, n1, x2, n2, s, mu0):
    """H1: logit p1 = m + d/2, logit p2 = m - d/2, d ~ N(0, s^2), m ~ N(mu0, 1). H0: d = 0."""
    def ll(m, d):
        p1, p2 = special.expit(m + d / 2), special.expit(m - d / 2)
        return stats.binom.logpmf(x1, n1, p1) + stats.binom.logpmf(x2, n2, p2)
    # centre the integration grid on the MLE for numerical stability
    mh = special.logit((x1 + x2 + 0.5) / (n1 + n2 + 1))
    c = ll(mh, 0.0)
    f0 = lambda m: np.exp(ll(m, 0.0) - c) * stats.norm.pdf(m, mu0, 1)
    m0 = integrate.quad(f0, mh - 3, mh + 3, limit=200, points=[mh])[0]
    f1 = lambda d, m: np.exp(ll(m, d) - c) * stats.norm.pdf(m, mu0, 1) * stats.norm.pdf(d, 0, s)
    m1 = integrate.dblquad(f1, mh - 3, mh + 3, lambda m: -3, lambda m: 3, epsabs=1e-12, epsrel=1e-7)[0]
    return float(m0 / m1)


def interpret(bf):
    if bf >= 1:
        k, side = bf, "H0 (no difference)"
    else:
        k, side = 1 / bf, "H1 (a difference)"
    lvl = ("anecdotal" if k < 3 else "moderate" if k < 10 else "strong" if k < 30 else "very strong" if k < 100 else "extreme")
    return f"{lvl} evidence for {side}"


def main():
    ev = load_events()
    pilot = pd.Timestamp(config()["splits"]["pilot_window_start"])
    rows = []
    for fam in ["broad", "strict"]:
        tv = tier1(ev, fam, ["train", "validation"])
        pi = float(tv["label_10"].mean())
        te = tier1(ev, fam, ["test"])
        for setname, d in [("full", te), ("without_pilot", te[te["date"] < pilot])]:
            for h in ["H1", "H2"]:
                if h == "H1":
                    dd = d[d["deliv_z20_prev"].notna()]
                    e = dd["deliv_z20_prev"] > 0
                else:
                    dd = d[d["ret_20"].notna() & d["oi_chg20"].notna()]
                    e = (dd["ret_20"] > 0) & (dd["oi_chg20"] < 0)
                x1, n1 = int(dd.loc[e, "label_10"].sum()), int(e.sum())
                x2, n2 = int(dd.loc[~e, "label_10"].sum()), int((~e).sum())
                priors = {"default Beta(1,1)": (1.0, 1.0), "wide Beta(0.5,0.5)": (0.5, 0.5),
                          "skeptical Beta(100*pi,100*(1-pi))": (100 * pi, 100 * (1 - pi))}
                for pname, (a, b) in priors.items():
                    bf = bf01_beta(x1, n1, x2, n2, a, b)
                    rows.append({"family": fam.upper(), "set": setname, "hypothesis": h, "model": "two-beta", "prior": pname,
                                 "x1": x1, "n1": n1, "x2": x2, "n2": n2, "diff_pp": 100 * (x1 / n1 - x2 / n2),
                                 "BF01": bf, "log10_BF01": float(np.log10(bf)), "interpretation": interpret(bf)})
                for s, pname in [(0.25, "skeptical logOR~N(0,0.25^2)"), (0.5, "medium logOR~N(0,0.5^2)"),
                                 (1.0, "default logOR~N(0,1)")]:
                    bf = bf01_logit(x1, n1, x2, n2, s, special.logit(pi))
                    rows.append({"family": fam.upper(), "set": setname, "hypothesis": h, "model": "logit-normal", "prior": pname,
                                 "x1": x1, "n1": n1, "x2": x2, "n2": n2, "diff_pp": 100 * (x1 / n1 - x2 / n2),
                                 "BF01": bf, "log10_BF01": float(np.log10(bf)), "interpretation": interpret(bf)})
    save({"task": "D Bayes factors", "method": __doc__.strip().splitlines()[2:17], "rows": rows,
          "caveat": "BF01 depends on the prior width: wide priors on the alternative favour H0 (Lindley effect). "
                    "The skeptical prior is the most informative check because it lets the alternative predict "
                    "realistic effects of a few pp."}, "D_bayes.json")
    print(pd.DataFrame(rows)[["family", "set", "hypothesis", "model", "prior", "diff_pp", "BF01", "interpretation"]].round(3).to_string())


if __name__ == "__main__":
    main()
