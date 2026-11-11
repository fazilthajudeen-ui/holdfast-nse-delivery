"""Shared helpers for the POST-HOC robustness suite (written 11 Nov 2026, after the test unlock of
10 Oct 2026 09:23:35 IST). Nothing here writes to the locked study files."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from impi.common import METRICS, RESULTS, RESULTS_LOG  # noqa: E402
from impi.pipeline.p4_composite import MODELS  # noqa: E402

OUTDIR = RESULTS / "posthoc"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR = ROOT / "paper" / "figs"
TABDIR = ROOT / "paper" / "tables"
LABEL = "POST-HOC (designed after the test set was opened on 10 Oct 2026 09:23:35 IST)"

LOCKED = [RESULTS_LOG, RESULTS / "test_lock.json", METRICS / "p6_test.json", METRICS / "p6_confirmatory_table.csv",
          METRICS / "feature_freeze.json", METRICS / "model_freeze.json", MODELS / "ml_frozen.pkl", MODELS / "composite.pkl"]


def checksums() -> dict:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in LOCKED}


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def _clean(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def save(obj: dict, name: str) -> Path:
    obj = {"label": LABEL, **obj}
    p = OUTDIR / name
    p.write_text(json.dumps(_clean(json.loads(json.dumps(obj, default=_default))), indent=2), encoding="utf-8")
    return p


def load(name: str) -> dict:
    return json.loads((OUTDIR / name).read_text(encoding="utf-8"))


def two_prop(x1, n1, x2, n2) -> dict:
    """Difference in proportions with unpooled (Wald) SE, pooled-z p-value (as the pre-registered test)."""
    from scipy import stats
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    se_u = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pp = (x1 + x2) / (n1 + n2)
    se_p = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = diff / se_p if se_p > 0 else np.nan
    return {"p1": p1, "p2": p2, "n1": int(n1), "n2": int(n2), "diff": diff, "se": se_u, "z": z,
            "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "ci95": [diff - 1.959964 * se_u, diff + 1.959964 * se_u],
            "ci90": [diff - 1.644854 * se_u, diff + 1.644854 * se_u]}


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # IEEE print rules: every label >= 8 pt at printed size (figures are drawn at their printed width),
    # Times-like STIX font, TrueType (Type 42) embedding - no Type 3 fonts.
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "standard"})
    return plt
