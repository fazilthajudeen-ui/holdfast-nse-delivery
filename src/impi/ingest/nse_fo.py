"""F&O bhavcopy parsers (legacy format to 2024-07-05, UDiFF common format from 2024-07-08)
and the per-underlying daily aggregation used by the positioning features.

Units: open interest is in shares (units) in both formats; this is verified on the
fo_overlap_check_dates in P0 QC. Contract counts are NOT used for stock positioning (D-004).
"""
from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd

from impi.ingest.nse_bhav import _num, _parse_date, _text, unzip

CONTRACT_COLS = ["date", "instrument", "symbol", "expiry", "strike", "opt_type", "open", "high", "low",
                 "close", "settle", "oi", "chg_oi", "lot", "underlying"]
UDIFF_TYPES = {"STF": "FUTSTK", "STO": "OPTSTK", "IDF": "FUTIDX", "IDO": "OPTIDX"}


def legacy_file_date(content: bytes) -> dt.date | None:
    try:
        lines = [l for l in _text(unzip(content)).splitlines() if l.strip()]
    except Exception:
        return None
    if len(lines) < 2 or not lines[0].startswith("INSTRUMENT,SYMBOL"):
        return None
    hdr = [h.strip() for h in lines[0].split(",")]
    return _parse_date(lines[1].split(",")[hdr.index("TIMESTAMP")], ["%d-%b-%Y"])


def udiff_file_date(content: bytes) -> dt.date | None:
    try:
        lines = [l for l in _text(unzip(content)).splitlines()[:3] if l.strip()]
    except Exception:
        return None
    if len(lines) < 2 or not lines[0].startswith("TradDt"):
        return None
    return _parse_date(lines[1].split(",")[0], ["%Y-%m-%d"])


def parse_legacy(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(unzip(content)), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df = df.apply(lambda s: s.str.strip())
    opt = df["OPTION_TYP"].where(df["OPTION_TYP"].isin(["CE", "PE"]))
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y"),
        "instrument": df["INSTRUMENT"], "symbol": df["SYMBOL"],
        "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y"),
        "strike": _num(df["STRIKE_PR"]), "opt_type": opt,
        "open": _num(df["OPEN"]), "high": _num(df["HIGH"]), "low": _num(df["LOW"]),
        "close": _num(df["CLOSE"]), "settle": _num(df["SETTLE_PR"]),
        "oi": _num(df["OPEN_INT"]), "chg_oi": _num(df["CHG_IN_OI"]),
        "lot": np.nan, "underlying": np.nan,
    })
    return out[CONTRACT_COLS]


def parse_udiff(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(unzip(content)), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda s: s.str.strip())
    df = df[df["FinInstrmTp"].isin(UDIFF_TYPES)]
    opt = df["OptnTp"].where(df["OptnTp"].isin(["CE", "PE"]))
    out = pd.DataFrame({
        "date": pd.to_datetime(df["TradDt"], format="%Y-%m-%d"),
        "instrument": df["FinInstrmTp"].map(UDIFF_TYPES), "symbol": df["TckrSymb"],
        "expiry": pd.to_datetime(df["XpryDt"], format="%Y-%m-%d"),
        "strike": _num(df["StrkPric"]), "opt_type": opt,
        "open": _num(df["OpnPric"]), "high": _num(df["HghPric"]), "low": _num(df["LwPric"]),
        "close": _num(df["ClsPric"]), "settle": _num(df["SttlmPric"]),
        "oi": _num(df["OpnIntrst"]), "chg_oi": _num(df["ChngInOpnIntrst"]),
        "lot": _num(df["NewBrdLotQty"]), "underlying": _num(df["UndrlygPric"]),
    })
    return out[CONTRACT_COLS]


def aggregate_stock(contracts: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, underlying stock): futures OI (sum over expiries), near/next-month
    futures prices and days to expiry, stock option call/put OI. Index contracts are dropped."""
    c = contracts[contracts["instrument"].isin(["FUTSTK", "OPTSTK"])].copy()
    if c.empty:
        return pd.DataFrame()
    fut = c[c["instrument"] == "FUTSTK"].sort_values(["symbol", "expiry"])
    g = fut.groupby(["date", "symbol"])
    agg = pd.DataFrame({
        "fut_oi": g["oi"].sum(min_count=1),
        "fut_chg_oi": g["chg_oi"].sum(min_count=1),
        "n_fut_expiries": g["expiry"].nunique(),
    })
    first = g.nth(0).set_index(["date", "symbol"])
    agg["near_expiry"] = first["expiry"]
    agg["near_settle"] = first["settle"]
    agg["near_close"] = first["close"]
    agg["near_oi"] = first["oi"]
    agg["lot"] = first["lot"]
    second = g.nth(1).set_index(["date", "symbol"])
    agg["next_expiry"] = second["expiry"]
    agg["next_settle"] = second["settle"]
    opt = c[c["instrument"] == "OPTSTK"]
    if not opt.empty:
        po = opt.pivot_table(index=["date", "symbol"], columns="opt_type", values="oi", aggfunc="sum")
        agg["call_oi"] = po.get("CE")
        agg["put_oi"] = po.get("PE")
    else:
        agg["call_oi"] = np.nan
        agg["put_oi"] = np.nan
    agg = agg.reset_index()
    return agg
