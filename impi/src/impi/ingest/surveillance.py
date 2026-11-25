"""Surveillance data: F&O ban list (available historically) and ASM/GSM (NOT available historically).

Ban file fo_secban_DDMMYYYY.csv lists securities in ban for that TRADE date; NSE publishes it the
previous trading day after close, so the ban for T+1 is known at the T decision point (G1).

ASM/GSM (D-009): NSE publishes only the current lists; no point-in-time history exists in the
official archives. The exclusion therefore cannot be applied historically. This is reported as a
limitation instead of being approximated with today's lists (which would add look-ahead).
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd

from impi.ingest.nse_bhav import _parse_date, _text

_RE = re.compile(r"Trade Date\s*(\d{1,2}-[A-Za-z]{3}-\d{4})")

ASM_GSM_AVAILABLE = False


def ban_file_date(content: bytes) -> dt.date | None:
    t = _text(content)
    m = _RE.search(t[:300])
    return _parse_date(m.group(1), ["%d-%b-%Y"]) if m else None


def parse_ban(content: bytes, date: dt.date) -> pd.DataFrame:
    """Rows: (date, symbol). An empty frame means NIL."""
    lines = [l.strip() for l in _text(content).splitlines() if l.strip()]
    syms = []
    for l in lines[1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1]:
            syms.append(parts[1])
        elif len(parts) == 1 and parts[0] and parts[0].upper() != "NIL":
            syms.append(parts[0])
    # Some files put the first symbol on the header line after the colon
    head = lines[0].split(":", 1)[1].strip() if lines and ":" in lines[0] else ""
    if head and head.upper() != "NIL":
        for p in [x.strip() for x in head.split(",") if x.strip()]:
            if not p.isdigit():
                syms.append(p)
    return pd.DataFrame({"date": pd.Timestamp(date), "symbol": sorted(set(syms))})
