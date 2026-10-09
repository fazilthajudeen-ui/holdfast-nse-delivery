"""Participant-wise open interest (contracts) in equity derivatives.

IMPORTANT (D-001): NSE publishes this only as MARKET-WIDE totals per participant category
(Client, DII, FII, Pro, TOTAL). There is no per-stock participant breakdown, so these series are
used as market-level regime features, never as stock-level positioning.
"""
from __future__ import annotations

import datetime as dt
import io
import re

import pandas as pd

from impi.ingest.nse_bhav import _num, _text

COLS = {
    "Future Index Long": "fut_idx_long", "Future Index Short": "fut_idx_short",
    "Future Stock Long": "fut_stk_long", "Future Stock Short": "fut_stk_short",
    "Option Index Call Long": "opt_idx_call_long", "Option Index Put Long": "opt_idx_put_long",
    "Option Index Call Short": "opt_idx_call_short", "Option Index Put Short": "opt_idx_put_short",
    "Option Stock Call Long": "opt_stk_call_long", "Option Stock Put Long": "opt_stk_put_long",
    "Option Stock Call Short": "opt_stk_call_short", "Option Stock Put Short": "opt_stk_put_short",
    "Total Long Contracts": "total_long", "Total Short Contracts": "total_short",
}
PARTICIPANTS = ["Client", "DII", "FII", "Pro", "TOTAL"]
_DATE_RE = re.compile(r"as on\s+([A-Za-z]+)\.?\s+(\d{1,2}),?\s*(\d{4})")


def file_date(content: bytes) -> dt.date | None:
    first = _text(content).splitlines()[0] if content else ""
    if "Participant wise Open Interest" not in first:
        return None
    m = _DATE_RE.search(first)
    if not m:
        return None
    s = f"{m.group(1)[:3]} {m.group(2)} {m.group(3)}"
    try:
        return dt.datetime.strptime(s, "%b %d %Y").date()
    except ValueError:
        return None


def parse(content: bytes, date: dt.date) -> pd.DataFrame:
    lines = _text(content).splitlines()
    body = "\n".join(lines[1:])
    if len(lines) > 1 and "Client Type" not in lines[1]:
        # some early files omit the header row; NSE's column order is fixed
        body = ",".join(["Client Type"] + list(COLS)) + "\n" + body
    df = pd.read_csv(io.StringIO(body), dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df["Client Type"] = df["Client Type"].str.strip()
    df = df[df["Client Type"].isin(PARTICIPANTS)]
    out = pd.DataFrame({"date": pd.Timestamp(date), "participant": df["Client Type"].values})
    for src, dst in COLS.items():
        out[dst] = _num(df[src]).values if src in df else float("nan")
    return out
