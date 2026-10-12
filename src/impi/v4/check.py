"""v4 Evaluation 1: the 2023-2026 check, run ONCE (pre_registered_v4.md).

    python -m impi.v4.check

Rules fixed in advance; the period was previously used for v1 (and post-hoc v2/v3), so results are
labelled "rules fixed in advance, previously used period". A lock file prevents a second run.
"""
from __future__ import annotations

import datetime as dt
import json
import sys

import pandas as pd

from impi.common import METRICS, RESULTS, git_commit, log, save_json
from impi.v4.signals import MARKET_V4, SIGNALS
from impi.v4.stats import run_all

LOCK = RESULTS / "v4_check_lock.json"
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-10-09")


def main():
    if LOCK.exists():
        sys.exit(f"v4 check already run ({json.loads(LOCK.read_text())['started']}); it may only run once.")
    LOCK.write_text(json.dumps({"started": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit()}),
                    encoding="utf-8")
    sig = pd.read_parquet(SIGNALS)
    sig = sig[sig["eligible"] & (sig["date"] >= START) & (sig["date"] <= END)]
    mk = pd.read_parquet(MARKET_V4)
    mk = mk[(mk["date"] >= START) & (mk["date"] <= END)]
    res = run_all(sig, mk)
    out = {"label": "rules fixed in advance (pre_registered_v4.md); previously used period - not a clean test",
           "window": [str(START.date()), str(END.date())], "results": res,
           "finished": dt.datetime.now().isoformat(timespec="seconds")}
    save_json(out, METRICS / "v4_check.json")
    for r in res:
        log(f"  {r['hypothesis']:4s} est {r.get('estimate')!s:>24} p {r.get('p')!s:>22} q {r.get('bh_q')!s:>22} "
            f"supported {r['supported']}  ({r.get('label')})")


if __name__ == "__main__":
    main()
