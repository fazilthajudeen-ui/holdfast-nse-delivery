"""Shared helpers: configuration, paths, logging, reproducibility (G7) and the results log (G5)."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]          # .../impi
CONFIG_DIR = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
RESULTS_LOG = RESULTS / "results_log.csv"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


@lru_cache(maxsize=1)
def config() -> dict:
    """All config/*.yaml files merged into one dict keyed by file stem, plus data/release_lag.yaml."""
    out = {}
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        with open(p, encoding="utf-8") as f:
            out[p.stem] = yaml.safe_load(f)
    lag = DATA / "release_lag.yaml"
    if lag.exists():
        with open(lag, encoding="utf-8") as f:
            out["release_lag"] = yaml.safe_load(f)
    return out


def config_hash() -> str:
    """Hash of every config file's exact bytes (recorded with every result, G7)."""
    h = hashlib.sha256()
    files = sorted(CONFIG_DIR.glob("*.yaml")) + [DATA / "release_lag.yaml"]
    for p in files:
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip() or "uncommitted"
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "config"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def rng(offset: int = 0) -> np.random.Generator:
    """Every random operation draws from a generator seeded from config (G7)."""
    return np.random.default_rng(int(config()["models"]["seed"]) + offset)


def seed(offset: int = 0) -> int:
    return int(config()["models"]["seed"]) + offset


RESULT_FIELDS = ["timestamp", "phase", "experiment", "variant", "split", "n", "metric", "value",
                 "details", "seed", "config_hash", "git_commit"]


def log_result(phase: str, experiment: str, variant: str, split: str, n: int, metric: str,
               value: float, details: dict | None = None, seed_value: int | None = None) -> None:
    """Append one row per variant tried (G5: nothing is silently discarded)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_LOG.exists()
    with open(RESULTS_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "phase": phase, "experiment": experiment, "variant": variant, "split": split,
            "n": int(n), "metric": metric,
            "value": "" if value is None or (isinstance(value, float) and np.isnan(value)) else float(value),
            "details": json.dumps(details or {}, default=str, sort_keys=True),
            "seed": "" if seed_value is None else seed_value,
            "config_hash": config_hash(), "git_commit": git_commit(),
        })


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def d(s) -> dt.date:
    """Parse a YAML date (already a date) or ISO string."""
    return s if isinstance(s, dt.date) else dt.date.fromisoformat(str(s))


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(config(), indent=1, default=str)[:2000]); print(config_hash(), git_commit(), sys.version)
