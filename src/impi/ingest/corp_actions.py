"""Reference data from NSE: corporate actions (split/bonus/consolidation/demerger), symbol changes
and results board-meeting dates (for the +/-3-day earnings exclusion).

Price factor convention: a split 10 -> 2 has factor 0.2. Prices BEFORE the ex-date are multiplied by
the product of later factors; quantities (volume, delivery qty, F&O OI in shares) are divided by it.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from impi.common import RAW

REF = RAW / "reference"
_BONUS = re.compile(r"bonus[^0-9]{0,20}(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", re.I)
# bonus of preference shares / NCRPS / debentures does not change the equity share count
_NON_EQUITY_BONUS = re.compile(r"preference|ncrps|\bncd|debenture|redeemable|warrant", re.I)
_FV_KEYWORD = re.compile(r"spli?t|sub-?division|consolidat|\bfv\b|face\s*val", re.I)   # incl. "Fv Splt Frm Rs 10 To Rs 2"
_FV = re.compile(r"(\d+(?:\.\d+)?)\D{0,30}?\bto\b\D{0,25}?(\d+(?:\.\d+)?)", re.I)
_DEMERGER = re.compile(r"demerger|spin.?off|scheme of arrangement|capital reduction", re.I)


def parse_subject(subject: str) -> tuple[str | None, float | None]:
    """('bonus'|'split'|'consolidation'|'bonus+split'|'demerger', price factor or None)."""
    s = (subject or "").strip()
    kind, factor = None, 1.0
    m = _BONUS.search(s)
    if m and not _NON_EQUITY_BONUS.search(s):
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and b > 0:
            factor *= b / (a + b)
            kind = "bonus"
    k = _FV_KEYWORD.search(s)
    if k:
        m2 = _FV.search(s[k.end():])
        if m2:
            old, new = float(m2.group(1)), float(m2.group(2))
            if old > 0 and new > 0 and old != new:
                factor *= new / old
                part = "consolidation" if new > old else "split"
                kind = part if kind is None else f"{kind}+{part}"
    if kind is None and _DEMERGER.search(s):
        return "demerger", None
    return (kind, factor) if kind else (None, None)


def load_corp_actions() -> pd.DataFrame:
    rows = []
    for p in sorted((REF / "corp_actions").glob("*.json")):
        js = json.loads(p.read_text(encoding="utf-8"))
        data = js.get("data", js) if isinstance(js, dict) else js
        for r in data or []:
            kind, factor = parse_subject(r.get("subject", ""))
            if kind is None:
                continue
            rows.append({"symbol": (r.get("symbol") or "").strip(), "series": (r.get("series") or "").strip(),
                         "ex_date": pd.to_datetime(r.get("exDate"), format="%d-%b-%Y", errors="coerce"),
                         "subject": r.get("subject", ""), "kind": kind, "factor": factor})
    df = pd.DataFrame(rows, columns=["symbol", "series", "ex_date", "subject", "kind", "factor"])
    df = df.dropna(subset=["ex_date"]).drop_duplicates(["symbol", "ex_date", "kind", "factor"])
    return df.sort_values(["ex_date", "symbol"]).reset_index(drop=True)


def load_symbol_changes() -> pd.DataFrame:
    p = REF / "symbolchange.csv"
    rows = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.rsplit(",", 3)
        if len(parts) != 4:
            continue
        old, new, date = parts[1].strip(), parts[2].strip(), parts[3].strip()
        dte = pd.to_datetime(date, format="%d-%b-%Y", errors="coerce")
        if old and new and old != new and pd.notna(dte):
            rows.append((old, new, dte))
    df = pd.DataFrame(rows, columns=["old", "new", "date"]).drop_duplicates()
    return df.sort_values("date").reset_index(drop=True)


def apply_symbol_changes(df: pd.DataFrame, changes: pd.DataFrame, date_col: str = "date",
                         sym_col: str = "symbol") -> tuple[pd.DataFrame, list[dict]]:
    """Rename an old symbol's rows dated BEFORE the change date to the new symbol, in date order
    (so A->B->C chains end at C). Skipped (and reported) if the new symbol already has rows before
    the change date (symbol reuse / conflict)."""
    out = df.copy()
    if isinstance(out[sym_col].dtype, pd.CategoricalDtype):
        out[sym_col] = out[sym_col].astype(str)
    applied = []
    present = set(out[sym_col].unique())
    for old, new, when in changes[["old", "new", "date"]].itertuples(index=False):
        if old not in present:
            continue
        mask_old = (out[sym_col] == old) & (out[date_col] < when)
        if not mask_old.any():
            continue
        if ((out[sym_col] == new) & (out[date_col] < when)).any():
            applied.append({"old": old, "new": new, "date": when, "status": "conflict_skipped"})
            continue
        out.loc[mask_old, sym_col] = new
        present.add(new)
        if not (out[sym_col] == old).any():
            present.discard(old)
        applied.append({"old": old, "new": new, "date": when, "status": "applied",
                        "rows": int(mask_old.sum())})
    return out, applied


_RESULTS = re.compile(r"financial result|results", re.I)


def load_results_dates() -> pd.DataFrame:
    """Board meetings whose purpose/description mentions results. known_at = intimation timestamp."""
    rows = []
    for p in sorted((REF / "board_meetings").glob("*.json")):
        js = json.loads(p.read_text(encoding="utf-8"))
        data = js.get("data", js) if isinstance(js, dict) else js
        for r in data or []:
            text = f"{r.get('bm_purpose', '')} {r.get('bm_desc', '')}"
            if not _RESULTS.search(text):
                continue
            rows.append({"symbol": (r.get("bm_symbol") or "").strip(),
                         "meeting_date": pd.to_datetime(r.get("bm_date"), format="%d-%b-%Y", errors="coerce"),
                         "known_at": pd.to_datetime(r.get("bm_timestamp"), format="%d-%b-%Y %H:%M:%S",
                                                    errors="coerce")})
    df = pd.DataFrame(rows, columns=["symbol", "meeting_date", "known_at"]).dropna(subset=["meeting_date"])
    df = df.sort_values("known_at").drop_duplicates(["symbol", "meeting_date"], keep="first")
    return df.sort_values(["meeting_date", "symbol"]).reset_index(drop=True)


def load_vix_history() -> pd.DataFrame:
    rows = []
    for p in sorted((REF / "vix_history").glob("*.json")):
        js = json.loads(p.read_text(encoding="utf-8"))
        data = js.get("data", js) if isinstance(js, dict) else js
        for r in data or []:
            rows.append({"date": pd.to_datetime(r.get("EOD_TIMESTAMP"), format="%d-%b-%Y", errors="coerce"),
                         "open": r.get("EOD_OPEN_INDEX_VAL"), "high": r.get("EOD_HIGH_INDEX_VAL"),
                         "low": r.get("EOD_LOW_INDEX_VAL"), "close": r.get("EOD_CLOSE_INDEX_VAL")})
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"]).dropna(subset=["date"])
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_sector_map() -> pd.DataFrame:
    """Current industry labels from NSE index constituent lists (D-008: no PIT sector history)."""
    frames = []
    for name in ["ind_niftytotalmarket_list.csv", "ind_nifty500list.csv"]:
        p = REF / name
        if p.exists():
            df = pd.read_csv(p, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            if {"Symbol", "Industry"} <= set(df.columns):
                frames.append(df[["Symbol", "Industry"]].rename(columns={"Symbol": "symbol", "Industry": "sector"}))
    if not frames:
        return pd.DataFrame(columns=["symbol", "sector"])
    out = pd.concat(frames).dropna()
    out["symbol"] = out["symbol"].str.strip()
    return out.drop_duplicates("symbol")
