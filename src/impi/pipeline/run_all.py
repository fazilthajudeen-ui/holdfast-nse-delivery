"""Run the IMPI phases in order.

    python -m impi.pipeline.run_all --to p2        # data -> QC -> G1 check -> features/events/freeze
    (write + commit pre_registered_hypotheses.md)
    python -m impi.pipeline.run_all --from p3 --to p5
    python -m impi.pipeline.p6_test                # the ONE test-set run (deliberately not chained)
    python -m impi.pipeline.run_all --from p7

P6 is never run by this script: touching the test set is a deliberate, separate step.
"""
from __future__ import annotations

import argparse
import importlib

from impi.common import log

STEPS = [
    ("download", "impi.ingest.download"),
    ("pit", "impi.transforms.pit_store"),
    ("adjust", "impi.transforms.adjust"),
    ("p0", "impi.pipeline.p0_qc"),
    ("p1", "impi.pipeline.p1_check"),
    ("p2", "impi.pipeline.p2_events"),
    ("p3", "impi.pipeline.p3_hypotheses"),
    ("p4", "impi.pipeline.p4_composite"),
    ("p5", "impi.pipeline.p5_ml"),
    ("p7", "impi.pipeline.p7_robustness"),
    ("p8", "impi.pipeline.p8_report"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="download")
    ap.add_argument("--to", dest="end", default="p8")
    a = ap.parse_args()
    names = [n for n, _ in STEPS]
    i, j = names.index(a.start), names.index(a.end)
    for name, mod in STEPS[i:j + 1]:
        log(f"===== {name} ({mod})")
        m = importlib.import_module(mod)
        m.main([]) if name == "download" else m.main()


if __name__ == "__main__":
    main()
