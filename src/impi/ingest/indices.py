"""Daily close of all NSE indices (ind_close_all): Nifty 50 / 200 / 500 and India VIX."""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from impi.common import config
from impi.ingest.nse_bhav import _num, _parse_date, _text

FMTS = ["%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y"]


def file_date(content: bytes) -> dt.date | None:
    lines = [l for l in _text(content).splitlines() if l.strip()]
    if len(lines) < 2 or not lines[0].startswith("Index Name"):
        return None
    p = lines[1].split(",")
    return _parse_date(p[1], FMTS) if len(p) > 1 else None


def parse(content: bytes, date: dt.date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["Index Name"] = df["Index Name"].str.strip()
    names = config()["data"]["index_names"]
    lookup = {n.lower(): key for key, alts in names.items() for n in alts}
    df["key"] = df["Index Name"].str.lower().map(lookup)
    df = df[df["key"].notna()].drop_duplicates("key")
    return pd.DataFrame({
        "date": pd.Timestamp(date), "index": df["key"].values,
        "open": _num(df["Open Index Value"]).values, "high": _num(df["High Index Value"]).values,
        "low": _num(df["Low Index Value"]).values, "close": _num(df["Closing Index Value"]).values,
    })
