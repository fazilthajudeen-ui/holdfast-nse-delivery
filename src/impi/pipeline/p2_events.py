"""P2 - feature library v1.0, events, labels, class balance and power calculation; then FREEZE.

    python -m impi.pipeline.p2_events            # builds features/events (slow) then summarises
    python -m impi.pipeline.p2_events --summary  # only re-summarise existing events

The freeze (results/metrics/feature_freeze.json) hashes features.yaml, labels.yaml, universe.yaml,
the FEATURES_V1 list and the source of every feature/label module. Later phases refuse to run if
any of them changed (spec: "Feature library v1.0 - freeze before any testing").
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from impi.common import CONFIG_DIR, METRICS, ROOT, log, save_json
from impi.models.ml import FEATURES_V1
from impi.research.stats import mde_auc_paired, mde_two_proportions
from impi.validation.splits import assign_split

FREEZE = METRICS / "feature_freeze.json"
FROZEN_FILES = ["config/features.yaml", "config/labels.yaml", "config/universe.yaml",
                "src/impi/features/delivery.py", "src/impi/features/volume.py", "src/impi/features/compression.py",
                "src/impi/features/vcp.py", "src/impi/features/positioning.py", "src/impi/features/divergence.py",
                "src/impi/features/regime.py", "src/impi/features/build.py", "src/impi/labels/triple_barrier.py",
                "src/impi/labels/vol_expansion.py", "src/impi/labels/events.py", "src/impi/transforms/adjust.py"]


def freeze_hash() -> dict:
    h = {}
    for f in FROZEN_FILES:
        data = (ROOT / f).read_bytes().replace(b"\r\n", b"\n")
        h[f] = hashlib.sha256(data).hexdigest()[:16]
    h["FEATURES_V1"] = hashlib.sha256(json.dumps(FEATURES_V1).encode()).hexdigest()[:16]
    return h


def assert_frozen() -> None:
    if not FREEZE.exists():
        raise SystemExit("Feature library not frozen yet - run P2 first.")
    saved = json.loads(FREEZE.read_text())["hashes"]
    now = freeze_hash()
    changed = [k for k in saved if saved[k] != now.get(k)]
    if changed:
        raise SystemExit(f"Frozen feature library changed after P2: {changed}. Re-run P2 and re-register.")


def load_events() -> pd.DataFrame:
    from impi.features.build import OUT
    ev = pd.read_parquet(OUT / "events.parquet")
    ev["split"] = assign_split(ev["date"], ev["label_end"])
    return ev


def summarise() -> dict:
    ev = load_events()
    t1 = ev[ev["tier1"] & ev["eligible"]]
    out = {"features_v1": len(FEATURES_V1),
           "missing_feature_columns": [f for f in FEATURES_V1 if f not in ev.columns]}
    counts = (t1.groupby(["family", "split"], dropna=False).size().unstack(fill_value=0))
    out["tier1_eligible_counts"] = {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in counts.to_dict("index").items()}
    out["tier1_all_counts"] = ev[ev["tier1"]].groupby("family").size().to_dict()
    t2 = ev[ev["tier2"] & ~ev["tier1"] & ev["eligible"]]
    out["tier2_only_eligible_counts"] = t2.groupby("family").size().to_dict()
    out["exclusions_tier1"] = {c: int(ev.loc[ev["tier1"], c].sum()) for c in
                               ["excl_ban", "excl_earnings", "excl_price", "excl_listing", "excl_history"]}
    bal = {}
    for fam, g in t1.groupby("family"):
        bal[fam] = {f"label_{h}": round(float(g[f"label_{h}"].mean()), 4) for h in (10, 20, 60)}
        bal[fam]["vol_exp"] = round(float(g["vol_exp_label"].mean()), 4)
        bal[fam]["labels_incomplete_10"] = int(g["label_10"].isna().sum())
    out["class_balance"] = bal
    # power: H1/H2 two-proportion MDE, H4 paired-AUC MDE, per family, train+val and test
    pw = {}
    for fam, g in t1[t1["label_10"].notna()].groupby("family"):
        for sp in ["trainval", "test"]:
            gg = g[g["split"].isin(["train", "validation"])] if sp == "trainval" else g[g["split"] == "test"]
            if len(gg) < 10:
                continue
            p = float(gg["label_10"].mean())
            e = gg["deliv_z20_prev"] > 0
            dv = (gg["ret_20"] > 0) & (gg["oi_chg20"] < 0)
            npos = int(gg["label_10"].sum())
            pw[f"{fam}_{sp}"] = {"n": len(gg), "base_rate": round(p, 4),
                                 "H1_mde_pp": round(100 * mde_two_proportions(p, int(e.sum()), int((~e).sum())), 2),
                                 "H2_mde_pp": round(100 * mde_two_proportions(p, int(dv.sum()), int((~dv).sum())), 2),
                                 "H4_mde_auc": round(mde_auc_paired(0.55, npos, len(gg) - npos), 4)}
    out["power"] = pw
    st = pw.get("strict_test", {})
    out["power_rule"] = ("Pre-registered rule: if the STRICT Tier-1 test set cannot detect a 10-pp H1 difference "
                         "(H1_mde_pp > 10), H1 is additionally tested on Tier-2 STRICT events (secondary); standards are "
                         "never lowered.")
    out["power_rule_triggered"] = bool(st.get("H1_mde_pp", 99) > 10)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if not a.summary:
        from impi.features import build
        build.main()
    s = summarise()
    save_json(s, METRICS / "p2_summary.json")
    save_json({"hashes": freeze_hash(), "features_v1": FEATURES_V1, "version": "1.0"}, FREEZE)
    log(f"P2 summary: {json.dumps(s['tier1_eligible_counts'])}; balance {s['class_balance']}; freeze written")


if __name__ == "__main__":
    main()
