"""Sub-period re-test on 2024-01-01 .. 2026-10-09 (requested by the team on 12 Oct 2026). POST-HOC / EXPLORATORY.

    python -m impi.research.subperiod

Re-runs the pre-registered v1 hypotheses (H1-H3 on STRICT and BROAD, H4 AUC and top-decile with the frozen
v1 models and composite) and the v4 hypotheses (D1, D2, H2', C1, V1) on the 2024-2026 part of the test period,
with exactly the same statistics and pass rule (expected direction AND BH q <= 0.10 within each family).

This does NOT replace the official results: the v1 verdicts come from the one-time 2023-2026 test (lock
results/test_lock.json, untouched here) and the v4 verdicts from its 2023-2026 check and the live forward
record. Choosing a sub-window after the full-period results are known is a forking path, so these numbers
are reported as exploratory only. Logged as phase "SUB-2024" in results_log.csv.
"""
from __future__ import annotations

import datetime as dt
import pickle

import numpy as np
import pandas as pd

from impi.common import METRICS, config, log, log_result, rng, save_json, seed
from impi.pipeline.common_phase import tier1
from impi.pipeline.p2_events import load_events
from impi.pipeline.p4_composite import MODELS
from impi.pipeline.p6_test import h4_tests
from impi.research.hypothesis_tests import amihud_buckets, run_family
from impi.validation.multiple_testing import benjamini_hochberg

START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-10-09")
LABEL = ("POST-HOC sub-period 2024-01-01..2026-10-09, requested after the full-period results were known; "
         "exploratory only - the official verdicts are the one-time 2023-26 test (v1) and the v4 check / live record")


def v1_family(ev: pd.DataFrame) -> list[dict]:
    with open(MODELS / "ml_frozen.pkl", "rb") as f:
        blob = pickle.load(f)
    with open(MODELS / "composite.pkl", "rb") as f:
        comp = pickle.load(f)
    cuts = amihud_buckets(tier1(ev, "broad", ["train"]))             # same liquidity buckets as the v1 test
    w = ev[(ev["date"] >= START) & (ev["date"] <= END)]
    fams = {"STRICT": tier1(w, "strict", ["test"]), "BROAD": tier1(w, "broad", ["test"])}
    tests = run_family(fams, cuts, rng(700))
    tests += h4_tests(fams["BROAD"], blob, comp, rng(701), int(config()["models"]["bootstrap"]["n"]))
    q, rej = benjamini_hochberg([t["p"] for t in tests], float(config()["models"]["fdr_q"]))
    for t, qq, rr in zip(tests, q, rej):
        t["bh_q"], t["bh_reject"] = qq, bool(rr)
        t["supported"] = bool(rr and t.get("direction_ok"))
        log_result("SUB-2024", "v1_family", f"{t['hypothesis']}_{t['family']}", "2024-2026", t["n"], "p", t["p"],
                   {"diff": t.get("diff"), "bh_q": qq}, seed_value=seed(700))
    return tests


def v4_family() -> list[dict]:
    from impi.v4.signals import MARKET_V4, SIGNALS
    from impi.v4.stats import run_all
    sig = pd.read_parquet(SIGNALS)
    sig = sig[sig["eligible"] & (sig["date"] >= START) & (sig["date"] <= END)]
    mk = pd.read_parquet(MARKET_V4)
    mk = mk[(mk["date"] >= START) & (mk["date"] <= END)]
    res = run_all(sig, mk)
    for r in res:
        log_result("SUB-2024", "v4_family", r["hypothesis"], "2024-2026", int(r.get("n_obs") or r.get("n1", 0) or 0),
                   "p", r.get("p"), {"estimate": r.get("estimate"), "bh_q": r.get("bh_q")}, seed_value=None)
    return res


def main():
    ev = load_events()
    out = {"label": LABEL, "window": [str(START.date()), str(END.date())],
           "v1": v1_family(ev), "v4": v4_family(), "finished": dt.datetime.now().isoformat(timespec="seconds")}
    save_json(out, METRICS / "subperiod_2024_2026.json")
    for t in out["v1"]:
        log(f"  v1 {t['hypothesis']:14s} {t['family']:6s} diff {t.get('diff')!s:>24} p {t.get('p')!s:>22} q {t.get('bh_q')!s:>22} supported {t['supported']}")
    for r in out["v4"]:
        log(f"  v4 {r['hypothesis']:14s}        est  {r.get('estimate')!s:>24} p {r.get('p')!s:>22} q {r.get('bh_q')!s:>22} supported {r['supported']}")


if __name__ == "__main__":
    main()
