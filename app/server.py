"""Holdfast (formerly Delivery Lens): website + chart-first terminal over the pipeline's output.

    python -m app.server            # starts on http://localhost:8000 and opens the browser
    python -m app.server --no-open  # server only

Reads data/derived/*.parquet and stats.json (written by the pipeline / daily update) and
reloads them automatically when the files change. Nothing here computes new results.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import webbrowser
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.common import DISCLAIMER, load_config, path
from src.regime import regimes, smooth_rel, zones

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Holdfast")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

from app.impi_api import FIGURES as IMPI_FIGURES, router as impi_router  # noqa: E402

app.include_router(impi_router)
if IMPI_FIGURES.exists():
    app.mount("/impi-figures", StaticFiles(directory=IMPI_FIGURES), name="impi-figures")


@app.middleware("http")
async def no_cache(request, call_next):
    """Always serve fresh page and data (the pipeline rewrites them every evening)."""
    resp = await call_next(request)
    if not request.url.path.startswith("/static/vendor"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def clean(v, nd=2):
    if v is None or v is pd.NA or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, float):
        return round(v, nd)
    return v


# ---------------------------------------------------------------- data cache
class Store:
    files = ("features", "scores", "events", "stats")

    def __init__(self):
        self.stamp = None

    def _stamp(self):
        return tuple(path(k).stat().st_mtime_ns if path(k).exists() else 0 for k in self.files)

    def get(self):
        st = self._stamp()
        if st != self.stamp:
            self.load()
            self.stamp = st
        return self

    def load(self):
        cfg = load_config()
        self.cfg = cfg
        self.feats = pd.read_parquet(path("features")).sort_values(["symbol", "date"])
        self.scores = pd.read_parquet(path("scores"))
        self.events = pd.read_parquet(path("events"))
        self.stats = json.loads(path("stats").read_text(encoding="utf-8"))
        self.as_of = self.scores["date"].max()
        self.names = {}
        lists = sorted(path("universe").glob("ind_nifty500list_*.csv"))
        if lists:
            u = pd.read_csv(lists[-1])
            u.columns = [c.strip() for c in u.columns]
            self.names = dict(zip(u["Symbol"].str.strip(), u["Company Name"].str.strip()))
        today = self.scores[self.scores["date"] == self.as_of].dropna(subset=["score"]).copy()
        prev = self.feats[self.feats["date"] < self.as_of].groupby("symbol")["close"].last()
        today["chg_1d"] = (today["close"] / today["symbol"].map(prev) - 1) * 100
        # yesterday's score per stock: lets the terminal open on today's freshest signal instead of the same leader
        prev_day = self.scores[self.scores["date"] < self.as_of]["date"].max()
        prev_score = self.scores[self.scores["date"] == prev_day].set_index("symbol")["score"]
        today["score_prev"] = today["symbol"].map(prev_score)
        # current buyer-zone state per stock (same rule as the chart)
        rg = cfg["regime"]
        zone_now = {}
        for sym, g in self.feats.groupby("symbol", sort=False):
            st = regimes(smooth_rel(g["rel_deliv"].reset_index(drop=True), rg["ema_span"]).to_numpy(),
                         rg["enter_pts"], rg["exit_pts"])
            z = zones(g["date"].reset_index(drop=True), st)
            zone_now[sym] = z[-1] if z and z[-1]["end"] == g["date"].iloc[-1].strftime("%Y-%m-%d") else None
        self.zone_now = zone_now
        self.spark = self.feats.groupby("symbol")["close"].apply(lambda s: [round(x, 2) for x in s.tail(20)]).to_dict()
        self.today = today.set_index("symbol")
        self.band_low, self.band_high = cfg["score"]["band_low"], cfg["score"]["band_high"]
        self.baseline_window = cfg["features"]["baseline_window"]
        self.history_days = cfg["score"]["history_days"]
        self.index = None
        if path("benchmark").exists():
            b = pd.read_parquet(path("benchmark")).sort_values("date")
            b = b[b["date"] <= self.as_of]
            if len(b) >= 2:
                self.index = {"name": "NIFTY 500", "close": round(float(b["nifty500"].iloc[-1]), 2),
                              "chg": round((b["nifty500"].iloc[-1] / b["nifty500"].iloc[-2] - 1) * 100, 2)}


STORE = Store()


def stock_detail(s: Store, sym: str) -> dict | None:
    df = s.feats[s.feats["symbol"] == sym]
    if df.empty:
        return None
    t = df["date"].dt.strftime("%Y-%m-%d")
    rg = s.cfg["regime"]
    sm_rel = smooth_rel(df["rel_deliv"].reset_index(drop=True), rg["ema_span"])
    sm_del = df["deliv_pct"].reset_index(drop=True).ewm(span=rg["ema_span"], adjust=False, min_periods=3).mean()
    state = regimes(sm_rel.to_numpy(), rg["enter_pts"], rg["exit_pts"])
    bars = [{"time": d, "open": round(o, 2), "high": round(h, 2), "low": round(l, 2), "close": round(c, 2),
             "vol": float(v) if pd.notna(v) else 0.0, "dq": float(q) if pd.notna(q) else None}
            for d, o, h, l, c, v, q in zip(t, df["open"], df["high"], df["low"], df["close"], df["volume"], df["deliv_qty"])]
    deliv = [{"time": d, "value": round(v, 1), "rel": None if pd.isna(b) else round(v - b, 1)}
             for d, v, b in zip(t, df["deliv_pct"], df["baseline_60"]) if pd.notna(v)]
    base = [{"time": d, "value": round(v, 1)} for d, v in zip(t, df["baseline_60"]) if pd.notna(v)]
    smooth = [{"time": d, "value": round(v, 1), "rel": None if pd.isna(r) else round(r, 1), "state": int(z)}
              for d, v, r, z in zip(t, sm_del, sm_rel, state) if pd.notna(v)]
    zone_list = zones(df["date"].reset_index(drop=True), state)
    current = zone_list[-1] if zone_list and zone_list[-1]["end"] == t.iloc[-1] else None
    ev = s.events[(s.events["symbol"] == sym) & (s.events["direction"] == "up")].sort_values("event_date")
    events = [{"date": r.event_date.strftime("%Y-%m-%d"), "start": r.start_date.strftime("%Y-%m-%d"),
               "move": clean(r.move_pct, 1), "deliv": clean(r.event_deliv, 1), "normal": clean(r.baseline_start, 1),
               "rel": clean(r.event_rel_deliv, 1), "group": r.group,
               "ret_60": clean(r.ret_60, 1), "exc_60": clean(r.exc_60, 1),
               "round_trip": None if pd.isna(r.round_trip) else bool(r.round_trip), "status": r.status}
              for r in ev.itertuples()]
    now = s.today.loc[sym] if sym in s.today.index else None
    method = None
    if now is not None:
        method = {"deliv": clean(now.deliv_pct, 1), "normal": clean(now.baseline_60, 1),
                  "rel_today": clean(now.rel_deliv, 1), "rel_5": clean(now.rel_deliv_5, 1),
                  "score": clean(now.score, 0), "band": clean(now.band),
                  "ret_20": clean(now.ret_20 * 100 if pd.notna(now.ret_20) else None, 1),
                  "chg_1d": clean(now.chg_1d), "close": clean(now.close)}
    return {"symbol": sym, "name": s.names.get(sym, ""), "bars": bars, "delivery": deliv, "baseline": base,
            "smooth": smooth, "zones": zone_list, "current_zone": current, "regime": rg,
            "events": events, "method": method}


# ---------------------------------------------------------------- pages
@app.get("/")
def home():
    return FileResponse(STATIC / "landing.html")


@app.get("/terminal")
def terminal():
    return FileResponse(STATIC / "terminal.html")


@app.get("/about")
def about():
    return FileResponse(STATIC / "about.html")


@app.get("/scanner")
def scanner_page():
    return FileResponse(STATIC / "scanner.html")


@app.get("/market")
def market_page():
    return FileResponse(STATIC / "market.html")


@app.get("/research")
def research_page():
    return FileResponse(STATIC / "research.html")


@app.get("/favicon.ico")
def favicon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28"><rect x="3" y="3" width="22" height="22" rx="7" '
           'fill="#fff" stroke="#16181D" stroke-width="2.6"/><rect x="8" y="13" width="12" height="7.5" rx="2.4" fill="#0071E3"/></svg>')
    return Response(svg, media_type="image/svg+xml")


# ---------------------------------------------------------------- API
@app.get("/api/meta")
def meta():
    s = STORE.get()
    up = s.stats["up"]
    g = {(r["group"], r["horizon"]): r for r in up["groups"]}
    rt = {r["group"]: r for r in up["round_trip"]["rows"]}
    hv = up["high_vs_low"]
    return {
        "as_of": s.as_of.strftime("%d %b %Y"),
        "n_stocks": int(len(s.today)),
        "median_delivery": int(round(float(s.feats["deliv_pct"].median()))),
        "n_days": int(s.feats["date"].nunique()),
        "index": s.index,
        "disclaimer": DISCLAIMER,
        "bands": {"low": s.band_low, "high": s.band_high},
        "method": {"baseline_days": s.baseline_window, "history_days": s.history_days},
        "evidence": {
            "n_events": up["n_events"], "n_stocks": up["n_stocks"],
            "diff_20": clean(hv["20"]["diff"], 1), "p_20": clean(hv["20"]["p"], 3),
            "diff_60": clean(hv["60"]["diff"], 1), "p_60": clean(hv["60"]["p"], 3),
            "beat_high": clean(g[("High", 60)]["hit_rate"], 0), "beat_low": clean(g[("Low", 60)]["hit_rate"], 0),
            "rt_high": clean(rt["High"]["rate"], 0), "rt_low": clean(rt["Low"]["rate"], 0),
            "exc20": {k: clean(g[(k, 20)]["mean_excess"], 2) for k in ("Low", "Mid", "High")},
        },
    }


@app.get("/api/watchlist")
def watchlist():
    s = STORE.get()
    t = s.today.reset_index()
    rows = [{"symbol": r.symbol, "name": s.names.get(r.symbol, ""), "close": clean(r.close),
             "chg_1d": clean(r.chg_1d), "ret_20": clean(r.ret_20 * 100 if pd.notna(r.ret_20) else None, 1),
             "deliv": clean(r.deliv_pct, 1), "normal": clean(r.baseline_60, 1),
             "score": clean(r.score, 1), "score_prev": clean(r.score_prev, 1), "band": clean(r.band),
             "spark": s.spark.get(r.symbol, []),
             "zone": (s.zone_now.get(r.symbol) or {}).get("kind")}
            for r in t.itertuples()]
    return {"as_of": s.as_of.strftime("%Y-%m-%d"), "rows": rows}


@app.get("/api/stock/{symbol}")
def stock(symbol: str):
    d = stock_detail(STORE.get(), symbol.upper())
    if d is None:
        raise HTTPException(404, f"{symbol.upper()} not found")
    return d


@app.post("/api/ask")
def ask(payload: dict = Body(...)):
    from app.assistant import answer
    s = STORE.get()
    return answer(s, str(payload.get("q", ""))[:500], payload.get("symbol"), lambda sym: stock_detail(s, sym))


def auto_update_loop(every_s: int = 600) -> None:
    """End-of-day auto update while the web app runs (backup for the scheduled task, e.g. when the laptop slept
    through 19:30). Every 10 minutes scripts/should_run.py decides; if an update is due (weekday after 19:30, last
    success older than that slot, no other run holding the lock) the nightly script starts hidden in the background.
    The pages reload themselves when the new trading day arrives (static/freshness.js)."""
    import subprocess
    import sys
    import time
    root = Path(__file__).resolve().parents[1]
    while True:
        try:
            r = subprocess.run([sys.executable, str(root / "scripts" / "should_run.py")], cwd=root, capture_output=True,
                               text=True, timeout=60)
            if r.returncode == 0:                    # due, and should_run.py has claimed the lock for this run
                with open(root / "data" / "logs" / "nightly.log", "a", encoding="utf-8") as f:
                    f.write(r.stdout.strip() + "  (started by the web app)\n")
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                # --force: should_run.py inside run_daily.bat would otherwise see our own lock and skip
                subprocess.Popen(["cmd", "/c", str(root / "scripts" / "run_daily.bat"), "--force"], cwd=root,
                                 creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("end-of-day update started in the background")
        except Exception as e:                     # never let the updater take the web app down
            print(f"auto-update check failed: {e}")
        time.sleep(every_s)


def main(argv=None):
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-auto-update", action="store_true", help="do not start the end-of-day update from the web app")
    a = ap.parse_args(argv)
    url = f"http://localhost:{a.port}"
    if not a.no_auto_update:
        threading.Thread(target=auto_update_loop, daemon=True, name="eod-auto-update").start()
    if not a.no_open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Holdfast on {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
