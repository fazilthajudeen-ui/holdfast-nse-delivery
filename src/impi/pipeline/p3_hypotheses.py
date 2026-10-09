"""P3 - H1-H3 event studies on TRAIN + VALIDATION only, with BH FDR (q = 0.10) across the
pre-registered family. The test set is not touched (G4).

    python -m impi.pipeline.p3_hypotheses
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.common import METRICS, config, log, log_result, rng, save_json, seed
from impi.pipeline.common_phase import assert_preregistered, tier1
from impi.pipeline.p2_events import assert_frozen, load_events
from impi.research.event_study import event_table
from impi.research.hypothesis_tests import amihud_buckets, run_family
from impi.validation.multiple_testing import benjamini_hochberg

SPLITS = ["train", "validation"]


def main():
    reg = assert_preregistered()
    assert_frozen()
    ev = load_events()
    q = float(config()["models"]["fdr_q"])
    fams = {"STRICT": tier1(ev, "strict", SPLITS), "BROAD": tier1(ev, "broad", SPLITS)}
    cuts = amihud_buckets(tier1(ev, "broad", ["train"]))
    r = rng(300)
    tests = run_family(fams, cuts, r)
    qv, rej = benjamini_hochberg([t["p"] for t in tests], q)
    for t, qq, rr in zip(tests, qv, rej):
        t["bh_q"] = qq
        t["bh_reject"] = bool(rr)
        t["supported"] = bool(rr and t.get("direction_ok"))
        log_result("P3", "hypothesis", f"{t['hypothesis']}_{t['family']}", "train+validation", t["n"], "p",
                   t["p"], {"diff": t["diff"], "bh_q": qq}, seed_value=seed(300))
    studies = {}
    for fam, d in fams.items():
        d = d.copy()
        d["h1_group"] = np.where(d["deliv_z20_prev"] > 0, "delivery_expansion", "no_expansion")
        studies[f"{fam}_by_delivery"] = event_table(d[d["deliv_z20_prev"].notna()], "h1_group", rng(301)).to_dict("records")
    out = {"registered": reg, "splits": SPLITS, "family_size": len(tests), "fdr_q": q, "tests": tests,
           "event_studies": studies,
           "n_events": {k: int(len(v)) for k, v in fams.items()}}
    save_json(out, METRICS / "p3_hypotheses.json")
    rows = [{"hypothesis": t["hypothesis"], "family": t["family"], "n_exposed": t.get("n1"), "n_rest": t.get("n2"),
             "rate_or_asym_exposed": t.get("p1", t.get("asym_exposed")), "rate_or_asym_rest": t.get("p2", t.get("asym_rest")),
             "diff": t["diff"], "p": t["p"], "bh_q": t["bh_q"], "expected": t["expected_sign"],
             "direction_ok": t["direction_ok"], "supported": t["supported"]} for t in tests]
    pd.DataFrame(rows).to_csv(METRICS / "p3_hypothesis_table.csv", index=False)
    log("P3 hypothesis table:\n" + pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
