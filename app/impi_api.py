"""IMPI endpoints for the Holdfast app.

Reads ONLY the files exported nightly by `python -m impi.live` (impi/data/live/app/*). Nothing here
computes predictions, so the web server needs no ML libraries. Files are reloaded when they change.

    GET /api/impi/meta              as-of date, model summary, live track record, availability
    GET /api/impi/scanner           today's row for every F&O stock (no explanations)
    GET /api/impi/stock/{symbol}    today's row + explanation + probability history + IMPI event markers
    GET /api/impi/market            participant positioning (FII/DII/Pro/Client) series + regime
    GET /api/impi/research          research results (pre-registered v1 + post-hoc v2) for the research page
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

IMPI = Path(__file__).resolve().parents[1] / "impi"
APP_DIR = IMPI / "data" / "live" / "app"
FIGURES = IMPI / "results" / "figures"
router = APIRouter(prefix="/api/impi")
FILES = ("meta.json", "research.json", "today.json", "history.parquet", "events.parquet", "market.parquet", "index.parquet",
         "v4_markers.parquet")


def _clean(v, nd=4):
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, nd)
    if isinstance(v, dict):
        return {k: _clean(x, nd) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x, nd) for x in v]
    return v


class ImpiStore:
    def __init__(self):
        self.stamp = None
        self.ok = False
        self.reason = "not loaded"
        self.lock = threading.Lock()

    def _stamp(self):
        return tuple((APP_DIR / f).stat().st_mtime_ns if (APP_DIR / f).exists() else 0 for f in FILES)

    def get(self):
        # FastAPI runs sync endpoints in a thread pool: the lock makes concurrent first requests wait for
        # the load instead of seeing a half-initialised store
        with self.lock:
            st = self._stamp()
            if st != self.stamp:
                try:
                    self.load()
                    self.ok, self.reason = True, ""
                except FileNotFoundError as e:
                    self.ok, self.reason = False, f"IMPI live data not built yet ({Path(e.filename).name}). Run: python -m impi.live"
                except Exception as e:                  # corrupt / partial export
                    self.ok, self.reason = False, f"IMPI live data could not be read: {type(e).__name__}"
                self.stamp = st
        return self

    def load(self):
        self.meta = json.loads((APP_DIR / "meta.json").read_text(encoding="utf-8"))
        self.research = json.loads((APP_DIR / "research.json").read_text(encoding="utf-8"))
        rows = json.loads((APP_DIR / "today.json").read_text(encoding="utf-8"))
        self.today = {r["symbol"]: r for r in rows}
        h = pd.read_parquet(APP_DIR / "history.parquet")
        h["date"] = pd.to_datetime(h["date"]).dt.strftime("%Y-%m-%d")
        self.history = {s: g.sort_values("date") for s, g in h.groupby("symbol")}
        e = pd.read_parquet(APP_DIR / "events.parquet")
        e["date"] = pd.to_datetime(e["date"]).dt.strftime("%Y-%m-%d")
        self.events = {s: g.sort_values("date") for s, g in e.groupby("symbol")}
        self.market = pd.read_parquet(APP_DIR / "market.parquet").sort_values("date")
        self.market["date"] = pd.to_datetime(self.market["date"]).dt.strftime("%Y-%m-%d")
        ix = pd.read_parquet(APP_DIR / "index.parquet")
        ix["date"] = pd.to_datetime(ix["date"]).dt.strftime("%Y-%m-%d")
        self.index = {k: g.sort_values("date") for k, g in ix.groupby("index")}
        self.v4 = {}
        vf = APP_DIR / "v4_markers.parquet"                     # optional (v4 signals, pre_registered_v4.md)
        if vf.exists():
            v = pd.read_parquet(vf)
            v["date"] = pd.to_datetime(v["date"]).dt.strftime("%Y-%m-%d")
            self.v4 = {s: g.sort_values("date") for s, g in v.groupby("symbol")}


STORE = ImpiStore()


def _need():
    s = STORE.get()
    if not s.ok:
        raise HTTPException(503, s.reason)
    return s


@router.get("/meta")
def meta():
    s = STORE.get()
    if not s.ok:
        return {"available": False, "reason": s.reason}
    return {"available": True, **_clean(s.meta)}


@router.get("/scanner")
def scanner():
    s = _need()
    rows = [{k: v for k, v in r.items() if k != "explain"} for r in s.today.values()]
    return {"as_of": s.meta["as_of"], "rows": _clean(rows)}


@router.get("/stock/{symbol}")
def stock(symbol: str):
    s = _need()
    sym = symbol.upper()
    row = s.today.get(sym)
    if row is None:
        return {"symbol": sym, "covered": False, "as_of": s.meta["as_of"]}
    h = s.history.get(sym)
    hist = [] if h is None else [
        {"time": d, "breakout": b, "outperform": o, "vol": v, "xs": x}
        for d, b, o, v, x in zip(h["date"], h["p_label_10"], h["p_label_xs10"], h["p_vol_exp"], h["xs_pct"])]
    ev = s.events.get(sym)
    events = [] if ev is None else [{"date": d, "family": f} for d, f in zip(ev["date"], ev["family"])]
    v4 = s.v4.get(sym)
    v4m = [] if v4 is None else [{"date": d, "kind": k} for d, k in zip(v4["date"], v4["kind"])]
    return {"symbol": sym, "covered": True, "as_of": s.meta["as_of"], "row": _clean(row),
            "history": _clean(hist), "events": events, "v4_markers": v4m, "models": _clean(s.meta.get("models", {})),
            "v4_check": _clean({r["hypothesis"]: {k: r.get(k) for k in ("estimate", "bh_q", "supported", "direction_ok")}
                                for r in ((s.research.get("v4_check") or {}).get("results") or [])})}


@router.get("/market")
def market():
    s = _need()
    m = s.market
    series = {}
    for p in ["fii", "dii", "pro", "client"]:
        for seg in ["fut_idx", "fut_stk"]:
            for k in ["share", "cot"]:
                c = f"{p}_{seg}_{k}"
                if c in m:
                    series[c] = [{"time": d, "value": round(float(v), 4)} for d, v in zip(m["date"], m[c]) if pd.notna(v)]
    last = m.iloc[-1].to_dict()
    regime = {k: _clean(last.get(k)) for k in ["date", "nifty_above_200", "nifty_trend_50_200", "vix_pct252", "breadth10",
                                               "fii_flow20", "dii_flow20", "nifty500_ret20"]}
    idx = {}
    for k in ["nifty50", "nifty500", "india_vix"]:
        g = s.index.get(k)
        if g is not None:
            idx[k] = [{"time": d, "value": round(float(v), 2)} for d, v in zip(g["date"], g["close"]) if pd.notna(v)]
    latest = {c: _clean(last.get(c)) for c in series}
    return {"as_of": s.meta["as_of"], "series": series, "latest": latest, "regime": regime, "index": idx,
            "v4_market": _clean(s.meta.get("v4_market")),
            "c1_check": _clean(next((r for r in ((s.research.get("v4_check") or {}).get("results") or []) if r.get("hypothesis") == "C1"), None))}


@router.get("/research")
def research():
    s = _need()
    figs = sorted(p.name for p in FIGURES.glob("*.png")) if FIGURES.exists() else []
    return {"as_of": s.meta["as_of"], **_clean(s.research), "track": _clean(s.meta.get("track")),
            "track_v3": _clean(s.meta.get("track_v3")), "track_v4": _clean(s.meta.get("track_v4")), "figures": figs}
