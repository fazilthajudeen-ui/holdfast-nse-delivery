"""Shared helpers: config loading, project paths, simple logging."""
from __future__ import annotations

import datetime as dt
import sys
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Research / education tool. Not investment advice."

# Windows consoles default to cp1252; make ₹ and other symbols printable.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for a key under `paths:` in config.yaml (parent dirs created)."""
    p = ROOT / load_config()["paths"][key]
    (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def append_log(name: str, line: str) -> None:
    """Append a line to data/logs/<name>."""
    with open(path("logs") / name, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
