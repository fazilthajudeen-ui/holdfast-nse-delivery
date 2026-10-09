"""Guards shared by phase runners: pre-registration committed, feature freeze intact, event subsets."""
from __future__ import annotations

import subprocess

import pandas as pd

from impi.common import ROOT, SANDBOX


def assert_preregistered() -> str:
    """The pre-registration must be committed to git and unchanged in the working copy (spec: write it
    BEFORE Phase 3). Returns the commit hash that registered it. Skipped only in the synthetic
    sandbox rehearsal (outputs redirected away from the real results)."""
    if SANDBOX:
        return "SANDBOX (synthetic rehearsal - not a real result)"
    f = "pre_registered_hypotheses.md"
    if not (ROOT / f).exists():
        raise SystemExit("pre_registered_hypotheses.md missing - write and commit it before P3.")
    log = subprocess.run(["git", "log", "--format=%H %cI", "--", f], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not log:
        raise SystemExit("pre_registered_hypotheses.md is not committed - commit it before P3.")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", f], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("pre_registered_hypotheses.md changed after registration - not allowed.")
    first = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H %cI", "--", f], cwd=ROOT,
                           capture_output=True, text=True).stdout.split()
    return " ".join(first[:2]) if first else " ".join(log[:2])


def tier1(ev: pd.DataFrame, family: str, splits: list[str]) -> pd.DataFrame:
    return ev[(ev["family"] == family) & ev["tier1"] & ev["eligible"] & ev["split"].isin(splits)
              & ev["label_10"].notna()].copy()


def tier2_only(ev: pd.DataFrame, family: str, splits: list[str]) -> pd.DataFrame:
    return ev[(ev["family"] == family) & ev["tier2"] & ~ev["tier1"] & ev["eligible"] & ev["split"].isin(splits)
              & ev["label_10"].notna()].copy()
