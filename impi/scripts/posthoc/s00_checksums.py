"""POST-HOC suite step 0: SHA-256 of every locked study file. Run with `before` first and `after` last.

    .venv/Scripts/python.exe scripts/posthoc/s00_checksums.py before|after
"""
import json
import sys

import pandas as pd

from _common import OUTDIR, checksums, save

when = sys.argv[1]
cs = checksums()
save({"when": when, "time": pd.Timestamp.now().isoformat(timespec="seconds"), "sha256": cs}, f"checksums_{when}.json")
if when == "after":
    before = json.loads((OUTDIR / "checksums_before.json").read_text())["sha256"]
    same = before == cs
    print("locked files unchanged:", same)
    for k in cs:
        print(("OK  " if before.get(k) == cs[k] else "DIFF"), k, cs[k][:16])
    sys.exit(0 if same else 1)
print(json.dumps({k: v[:16] for k, v in cs.items()}, indent=1))
