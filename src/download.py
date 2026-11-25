"""Download NSE daily "full bhavcopy + delivery" files (sec_bhavdata_full) and cache them.

Usage:
    python -m src.download --months 14 --source nse
    python -m src.download --start 2026-09-01 --end 2026-09-05
    python -m src.download --source github --start 2025-01-01 --end 2025-12-31

Politeness: browser User-Agent, cookie warm-up on nseindia.com, >= 1 s between requests,
retries with exponential backoff, and a cached file is never downloaded again.
Holidays: NSE answers 404 or a small HTML page -> the response is validated (must start
with "SYMBOL") and the date is remembered in data/logs/no_data_dates.txt so it is not
asked for again. Today's date is never marked as a holiday (it may just be unpublished).
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import pandas as pd
import requests

from src.common import append_log, load_config, log, path

NSE_HOME = "https://www.nseindia.com"
NSE_BHAV = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
NSE_INDEX = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{ddmmyyyy}.csv"
GITHUB_BHAV = ("https://raw.githubusercontent.com/chartiny/nse-sec-bhavdata-full/master/"
               "{yyyy}/nse-sec-bhavdata-full-{iso}.csv")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

NO_DATA_FILE = "no_data_dates.txt"


# ---------------------------------------------------------------- validation helpers
def is_valid_bhav(text: str) -> bool:
    """A real bhavcopy starts with the SYMBOL header; holiday pages are HTML."""
    return text.lstrip("﻿ \r\n\t").startswith("SYMBOL")


def is_valid_index(text: str) -> bool:
    return text.lstrip("﻿ \r\n\t").startswith("Index Name")


def file_date(text: str, kind: str = "bhav") -> dt.date | None:
    """Trading date written inside the file (DATE1 / Index Date of the first data row).
    NSE sometimes serves the previous day's file on a holiday, so this must match the URL date."""
    lines = [l for l in text.lstrip("﻿").splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    try:
        if kind == "bhav":
            return dt.datetime.strptime(lines[1].split(",")[2].strip(), "%d-%b-%Y").date()
        return dt.datetime.strptime(lines[1].split(",")[1].strip(), "%d-%m-%Y").date()
    except (IndexError, ValueError):
        return None


def verify_cache() -> int:
    """Remove cached files whose internal date differs from their file name (stale holiday copies)."""
    removed = 0
    for kind, folder, pattern in [("bhav", path("raw"), "sec_bhavdata_full_*.csv"),
                                  ("index", path("index"), "ind_close_all_*.csv")]:
        for p in folder.glob(pattern):
            want = dt.date.fromisoformat(p.stem.rsplit("_", 1)[1])
            with open(p, encoding="utf-8", errors="ignore") as f:
                head = f.readline() + f.readline()
            got = file_date(head, kind)
            if got != want:
                p.unlink()
                if kind == "bhav":
                    append_log(NO_DATA_FILE, want.isoformat())
                log(f"removed stale {p.name}: contains {got} data (holiday)")
                removed += 1
    return removed


def bhav_path(d: dt.date):
    return path("raw") / f"sec_bhavdata_full_{d:%Y-%m-%d}.csv"


def index_path(d: dt.date):
    return path("index") / f"ind_close_all_{d:%Y-%m-%d}.csv"


def cached_dates() -> list[dt.date]:
    out = []
    for p in path("raw").glob("sec_bhavdata_full_*.csv"):
        try:
            out.append(dt.date.fromisoformat(p.stem.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return sorted(out)


def no_data_dates() -> set[str]:
    f = path("logs") / NO_DATA_FILE
    return set(f.read_text().split()) if f.exists() else set()


# ---------------------------------------------------------------- HTTP
class Fetcher:
    def __init__(self, source: str):
        cfg = load_config()["download"]
        self.source = source
        self.sleep = max(1.0, float(cfg["sleep_seconds"]))
        self.retries = int(cfg["retries"])
        self.backoff = float(cfg["backoff_seconds"])
        self.timeout = float(cfg["timeout_seconds"])
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last = 0.0
        self.consecutive_blocks = 0
        if source == "nse":
            self.warm_up()

    def _wait(self):
        gap = time.monotonic() - self._last
        if gap < self.sleep:
            time.sleep(self.sleep - gap)
        self._last = time.monotonic()

    def warm_up(self):
        """GET the NSE home page so the session carries the cookies NSE expects."""
        try:
            self._wait()
            r = self.session.get(NSE_HOME, timeout=self.timeout)
            log(f"warm-up {NSE_HOME} -> HTTP {r.status_code}, cookies: {len(self.session.cookies)}")
        except requests.RequestException as e:
            log(f"warm-up failed ({e}); continuing without cookies")

    def get(self, url: str, validator) -> tuple[str, bytes | None]:
        """Returns (status, content). status: ok | missing | blocked | error."""
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                r = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as e:
                status, detail = "error", str(e)[:120]
            else:
                if r.status_code == 200:
                    text = r.content[:200].decode("utf-8", errors="ignore")
                    self.consecutive_blocks = 0
                    if validator(text):
                        return "ok", r.content
                    return "missing", None          # HTML holiday page served as 200
                if r.status_code == 404:
                    self.consecutive_blocks = 0
                    return "missing", None
                status, detail = ("blocked" if r.status_code in (401, 403) else "error"), f"HTTP {r.status_code}"
            if attempt < self.retries:
                wait = self.backoff * 2 ** attempt
                log(f"  {detail} for {url.rsplit('/', 1)[-1]}; retry {attempt + 1} in {wait:.0f}s")
                time.sleep(wait)
                if status == "blocked" and self.source == "nse":
                    self.warm_up()                  # cookies may have expired
        if status == "blocked":
            self.consecutive_blocks += 1
        return status, None


# ---------------------------------------------------------------- download jobs
def download_range(start: dt.date, end: dt.date, source: str, fetch_index: bool | None = None) -> dict:
    """Download every weekday in [start, end]. Returns counts per status."""
    cfg = load_config()
    fetch_index = cfg["download"]["fetch_index"] if fetch_index is None else fetch_index
    fetch_index = fetch_index and source == "nse"      # index files exist only on NSE
    today = dt.date.today()
    verify_cache()                 # drop stale holiday copies before building the skip list
    skip = no_data_dates()
    fetcher: Fetcher | None = None
    counts = {"ok": 0, "cached": 0, "missing": 0, "blocked": 0, "error": 0, "index_ok": 0}

    extra = {dt.date.fromisoformat(x) for x in cfg["download"].get("extra_dates", [])}
    days = sorted({d.date() for d in pd.bdate_range(start, end)} | {d for d in extra if start <= d <= end})
    log(f"{len(days)} weekdays from {start} to {end} (source={source})")
    for i, d in enumerate(days, 1):
        iso = d.isoformat()
        need_bhav = not bhav_path(d).exists() and iso not in skip
        need_index = fetch_index and not index_path(d).exists() and iso not in skip
        if not need_bhav and not need_index:
            counts["cached" if bhav_path(d).exists() else "missing"] += 1
            continue
        fetcher = fetcher or Fetcher(source)

        if need_bhav:
            ddmmyyyy = f"{d:%d%m%Y}"
            url = (NSE_BHAV.format(ddmmyyyy=ddmmyyyy) if source == "nse"
                   else GITHUB_BHAV.format(yyyy=d.year, iso=iso))
            status, content = fetcher.get(url, is_valid_bhav)
            if status == "ok" and file_date(content[:600].decode("utf-8", "ignore")) != d:
                status, content = "missing", None       # previous day's file served on a holiday
            if status == "ok":
                bhav_path(d).write_bytes(content)
            elif status == "missing" and d < today:
                append_log(NO_DATA_FILE, iso)       # weekday holiday: never ask again
            counts[status] += 1
            append_log("download_log.csv",
                       f"{dt.datetime.now():%Y-%m-%d %H:%M:%S},{iso},bhav,{source},{status},"
                       f"{len(content) if content else 0}")
            log(f"[{i}/{len(days)}] {iso} bhav: {status}")
            if status != "ok":
                need_index = False
            if fetcher.consecutive_blocks >= 3:
                log("NSE is refusing requests (3 blocked days in a row). "
                    "Try again later or use --source github.")
                break

        if need_index:
            status, content = fetcher.get(NSE_INDEX.format(ddmmyyyy=f"{d:%d%m%Y}"), is_valid_index)
            if status == "ok" and file_date(content[:600].decode("utf-8", "ignore"), "index") != d:
                status, content = "missing", None
            if status == "ok":
                index_path(d).write_bytes(content)
                counts["index_ok"] += 1
            append_log("download_log.csv",
                       f"{dt.datetime.now():%Y-%m-%d %H:%M:%S},{iso},index,{source},{status},"
                       f"{len(content) if content else 0}")
    log(f"done: {counts}")
    return counts


def download_universe(force: bool = False):
    """Save a dated copy of the Nifty 500 constituents list (once per day)."""
    cfg = load_config()["universe"]
    out = path("universe") / f"ind_nifty500list_{dt.date.today():%Y-%m-%d}.csv"
    if out.exists() and not force:
        return out
    fetcher = Fetcher("nse")
    status, content = fetcher.get(cfg["nifty500_url"], lambda t: "Symbol" in t)
    if status == "ok":
        out.write_bytes(content)
        log(f"saved Nifty 500 list -> {out.name}")
        return out
    log(f"Nifty 500 list unavailable ({status}); clean.py will fall back to liquidity filter")
    return None


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=dt.date.fromisoformat)
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date.today())
    ap.add_argument("--months", type=int, default=cfg["months"])
    ap.add_argument("--source", choices=["nse", "github"], default=cfg["source"])
    ap.add_argument("--no-index", action="store_true", help="skip the Nifty 500 index files")
    ap.add_argument("--no-universe", action="store_true", help="skip the Nifty 500 list")
    a = ap.parse_args(argv)

    start = a.start or (pd.Timestamp(a.end) - pd.DateOffset(months=a.months)).date()
    if not a.no_universe and a.source == "nse":
        download_universe()
    counts = download_range(start, a.end, a.source, fetch_index=False if a.no_index else None)
    if counts["ok"] + counts["cached"] == 0 and counts["blocked"]:
        raise SystemExit("No files could be downloaded from NSE. Re-run with --source github.")


if __name__ == "__main__":
    main()
