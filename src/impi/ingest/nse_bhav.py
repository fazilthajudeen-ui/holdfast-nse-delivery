"""Equity bhavcopy + delivery parsers.

Two eras (D-005):
  * sec_bhavdata_full (from 2019-09-30): prices, volume, turnover, trades, delivery in one file.
  * before that: legacy cm bhavcopy (prices, volume, value, trades, ISIN) joined with the MTO
    security-wise delivery file (deliverable quantity) on SYMBOL + SERIES.
Both are normalised to the same columns (EQUITY_COLS). The two sources are cross-checked on
overlap dates in P0 QC.
"""
from __future__ import annotations

import datetime as dt
import io
import zipfile

import numpy as np
import pandas as pd

EQUITY_COLS = ["date", "symbol", "series", "open", "high", "low", "close", "last", "prev_close",
               "volume", "turnover_rs", "trades", "deliv_qty", "deliv_pct", "isin", "source"]


def _text(content: bytes) -> str:
    return content.decode("utf-8", "ignore").lstrip("﻿")


def unzip(content: bytes) -> bytes:
    if content[:2] != b"PK":
        return content
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        return z.read(z.namelist()[0])


def _parse_date(s: str, fmts) -> dt.date | None:
    s = s.strip().strip('"')
    for f in fmts:
        try:
            return dt.datetime.strptime(s, f).date()
        except ValueError:
            continue
    return None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"-": np.nan, "": np.nan, "nan": np.nan}),
                         errors="coerce")


# ------------------------------------------------------------------ sec_bhavdata_full
def sec_file_date(content: bytes) -> dt.date | None:
    lines = [l for l in _text(content).splitlines() if l.strip()]
    if len(lines) < 2 or not lines[0].lstrip().startswith("SYMBOL"):
        return None
    parts = lines[1].split(",")
    return _parse_date(parts[2], ["%d-%b-%Y"]) if len(parts) > 2 else None


def parse_sec(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), dtype=str, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda s: s.str.strip())
    out = pd.DataFrame({
        "date": pd.to_datetime(df["DATE1"], format="%d-%b-%Y"),
        "symbol": df["SYMBOL"], "series": df["SERIES"],
        "open": _num(df["OPEN_PRICE"]), "high": _num(df["HIGH_PRICE"]), "low": _num(df["LOW_PRICE"]),
        "close": _num(df["CLOSE_PRICE"]), "last": _num(df["LAST_PRICE"]),
        "prev_close": _num(df["PREV_CLOSE"]),
        "volume": _num(df["TTL_TRD_QNTY"]), "turnover_rs": _num(df["TURNOVER_LACS"]) * 1e5,
        "trades": _num(df["NO_OF_TRADES"]),
        "deliv_qty": _num(df["DELIV_QTY"]), "deliv_pct": _num(df["DELIV_PER"]),
        "isin": None, "source": "sec_bhav",
    })
    return out[EQUITY_COLS]


# ------------------------------------------------------------------ legacy cm bhavcopy
def cm_file_date(content: bytes) -> dt.date | None:
    try:
        lines = [l for l in _text(unzip(content)).splitlines() if l.strip()]
    except zipfile.BadZipFile:
        return None
    if len(lines) < 2 or not lines[0].startswith("SYMBOL,SERIES,OPEN"):
        return None
    hdr = [h.strip() for h in lines[0].split(",")]
    i = hdr.index("TIMESTAMP")
    return _parse_date(lines[1].split(",")[i], ["%d-%b-%Y"])


def parse_cm(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(unzip(content)), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df = df.apply(lambda s: s.str.strip())
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y"),
        "symbol": df["SYMBOL"], "series": df["SERIES"],
        "open": _num(df["OPEN"]), "high": _num(df["HIGH"]), "low": _num(df["LOW"]),
        "close": _num(df["CLOSE"]), "last": _num(df["LAST"]), "prev_close": _num(df["PREVCLOSE"]),
        "volume": _num(df["TOTTRDQTY"]), "turnover_rs": _num(df["TOTTRDVAL"]),
        "trades": _num(df["TOTALTRADES"]) if "TOTALTRADES" in df else np.nan,
        "isin": df["ISIN"] if "ISIN" in df else None,
    })
    return out


# ------------------------------------------------------------------ MTO delivery file
def mto_file_date(content: bytes) -> dt.date | None:
    lines = [l for l in _text(content).splitlines() if l.strip()]
    if not lines or not lines[0].startswith("Security Wise Delivery Position"):
        return None
    for l in lines[1:4]:
        p = l.split(",")
        if len(p) > 2 and p[0].strip() == "10" and p[1].strip() == "MTO":
            return _parse_date(p[2], ["%d%m%Y"])
    return None


def parse_mto(content: bytes) -> pd.DataFrame:
    rows = []
    for l in _text(content).splitlines():
        p = [x.strip() for x in l.split(",")]
        if len(p) >= 7 and p[0] == "20":
            rows.append((p[2], p[3], p[4], p[5], p[6]))
    df = pd.DataFrame(rows, columns=["symbol", "series", "mto_volume", "deliv_qty", "deliv_pct"])
    for c in ["mto_volume", "deliv_qty", "deliv_pct"]:
        df[c] = _num(df[c])
    return df


def merge_legacy(cm: pd.DataFrame, mto: pd.DataFrame | None) -> pd.DataFrame:
    """Join prices with delivery. Delivery is kept only when MTO's traded quantity equals the
    bhavcopy volume (guards against mismatched rows)."""
    out = cm.copy()
    if mto is None or mto.empty:
        out["deliv_qty"] = np.nan
        out["deliv_pct"] = np.nan
    else:
        m = mto.drop_duplicates(["symbol", "series"])
        out = out.merge(m, on=["symbol", "series"], how="left")
        bad = out["mto_volume"].notna() & (out["mto_volume"] != out["volume"])
        out.loc[bad, ["deliv_qty", "deliv_pct"]] = np.nan
        out = out.drop(columns=["mto_volume"])
    out["source"] = "cm_legacy+mto"
    return out[EQUITY_COLS]
