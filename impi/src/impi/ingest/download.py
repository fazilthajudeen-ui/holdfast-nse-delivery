"""Download every raw dataset IMPI needs from official NSE archives, politely and resumably.

    python -m impi.ingest.download                 # reference data + full daily range from config
    python -m impi.ingest.download --start 2015-06-01 --end 2015-06-05
    python -m impi.ingest.download --reference-only

Rules:
  * each file is validated: right header AND the trade date written inside the file must equal the
    requested date (NSE serves the previous day's file on some holidays -> 'stale');
  * a date with no equity file is a non-trading day: nothing else is requested for it;
  * every file is cached and never downloaded twice; 404s/stale answers are remembered in
    data/raw/_meta/no_data.csv; network errors are NOT remembered, so a re-run retries them;
  * >= 1 s between requests, retries with exponential backoff, stop after repeated blocks.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import time
from pathlib import Path

from impi.common import RAW, ROOT, config, d as as_date, log
from impi.ingest import indices, nse_bhav, nse_fo, participant, surveillance
from impi.ingest.http import Fetcher

VALIDATORS = {
    "sec_bhav": nse_bhav.sec_file_date,
    "cm_legacy": nse_bhav.cm_file_date,
    "mto": nse_bhav.mto_file_date,
    "fo_legacy": nse_fo.legacy_file_date,
    "fo_udiff": nse_fo.udiff_file_date,
    "participant_oi": participant.file_date,
    "ban": surveillance.ban_file_date,
    "index_close": indices.file_date,
}
EXT = {"sec_bhav": "csv", "cm_legacy": "zip", "mto": "DAT", "fo_legacy": "zip", "fo_udiff": "zip",
       "participant_oi": "csv", "ban": "csv", "index_close": "csv"}
META = RAW / "_meta"
NO_DATA = META / "no_data.csv"
DL_LOG = META / "download_log.csv"
RECENT_DAYS = 3                 # dates this close to today are never recorded as "no data"


def url_for(name: str, d: dt.date) -> str:
    mon = d.strftime("%b").upper()
    return config()["data"]["sources"][name]["url"].format(
        dd=f"{d:%d}", mm=f"{d:%m}", yyyy=f"{d:%Y}", MON=mon, ddmmyyyy=f"{d:%d%m%Y}",
        yyyymmdd=f"{d:%Y%m%d}")


def raw_path(name: str, d: dt.date) -> Path:
    return RAW / name / f"{d:%Y}" / f"{name}_{d:%Y-%m-%d}.{EXT[name]}"


def _append(path: Path, row: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def load_no_data() -> set[tuple[str, str]]:
    if not NO_DATA.exists():
        return set()
    with open(NO_DATA, encoding="utf-8") as f:
        return {(r[0], r[1]) for r in csv.reader(f) if len(r) >= 2}


class Downloader:
    def __init__(self):
        self.f: Fetcher | None = None
        self.no_data = load_no_data()
        self.counts = {"ok": 0, "cached": 0, "missing": 0, "stale": 0, "error": 0, "blocked": 0}

    def fetcher(self) -> Fetcher:
        if self.f is None:
            self.f = Fetcher()
        return self.f

    def mark_no_data(self, iso: str, name: str, reason: str) -> None:
        # files for the last few days may simply not be published yet (the nightly run starts ~19:30):
        # never remember them as missing, so the next run retries them
        if (dt.date.today() - dt.date.fromisoformat(iso)).days <= RECENT_DAYS:
            return
        if (iso, name) not in self.no_data:
            self.no_data.add((iso, name))
            _append(NO_DATA, [iso, name, reason])

    def fetch(self, name: str, d: dt.date) -> str:
        p, iso = raw_path(name, d), d.isoformat()
        if p.exists():
            self.counts["cached"] += 1
            return "cached"
        if (iso, name) in self.no_data:
            return "missing"
        url = url_for(name, d)
        status, content = self.fetcher().get(url)
        if status == "ok":
            got = VALIDATORS[name](content)
            if got != d:
                status = "stale" if got else "missing"
                self.mark_no_data(iso, name, f"{status}:{got}")
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(p.suffix + ".part")
                tmp.write_bytes(content)
                os.replace(tmp, p)
        elif status == "missing":
            self.mark_no_data(iso, name, "404/html")
        self.counts[status] += 1
        _append(DL_LOG, [dt.datetime.now().isoformat(timespec="seconds"), iso, name, status,
                         len(content) if content else 0])
        if status in ("error", "blocked"):
            log(f"  {status}: {name} {iso}")
        return status

    # ------------------------------------------------------------ one trading date
    def fetch_date(self, d: dt.date) -> str:
        """Returns 'trading' | 'holiday' | 'error'."""
        c = config()["data"]
        iso = d.isoformat()
        if (iso, "equity") in self.no_data:
            return "holiday"
        src = c["sources"]
        eq_ok, errors = False, False
        if d >= as_date(src["sec_bhav"]["first_date"]):
            s = self.fetch("sec_bhav", d)
            eq_ok = s in ("ok", "cached")
            errors |= s in ("error", "blocked")
        if not eq_ok and not errors:
            s = self.fetch("cm_legacy", d)
            eq_ok = s in ("ok", "cached")
            errors |= s in ("error", "blocked")
            if eq_ok:
                self.fetch("mto", d)
        if not eq_ok:
            if not errors:
                self.mark_no_data(iso, "equity", "holiday")
                return "holiday"
            return "error"
        if d in {as_date(x) for x in c["overlap_check_dates"]}:
            self.fetch("cm_legacy", d)
            self.fetch("mto", d)
        # F&O bhavcopy: format by date, with fallback to the other format
        order = ["fo_legacy", "fo_udiff"] if d <= as_date(src["fo_legacy"]["last_date"]) else ["fo_udiff", "fo_legacy"]
        if d in {as_date(x) for x in c["fo_overlap_check_dates"]}:
            self.fetch("fo_legacy", d)
            self.fetch("fo_udiff", d)
        elif self.fetch(order[0], d) not in ("ok", "cached"):
            self.fetch(order[1], d)
        for name in ["participant_oi", "ban", "index_close"]:
            self.fetch(name, d)
        return "trading"


def candidate_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    special = {as_date(x) for x in config()["data"]["special_sessions"]}
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 or d in special:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def reuse_delivery_lens_cache() -> int:
    """Copy files the Delivery Lens project already downloaded from the same NSE URLs."""
    n = 0
    reuse = config()["data"].get("reuse_cache", {})
    patterns = {"sec_bhav": "sec_bhavdata_full_*.csv", "index_close": "ind_close_all_*.csv"}
    for name, rel in reuse.items():
        folder = (ROOT / rel).resolve()
        if not folder.exists():
            continue
        for p in folder.glob(patterns[name]):
            try:
                d = dt.date.fromisoformat(p.stem.rsplit("_", 1)[1])
            except ValueError:
                continue
            dst = raw_path(name, d)
            if dst.exists():
                continue
            if VALIDATORS[name](p.read_bytes()) != d:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
            n += 1
    return n


# ------------------------------------------------------------ reference data (APIs, lists)
def download_reference(dl: Downloader, start: dt.date, end: dt.date) -> None:
    c = config()["data"]["sources"]
    ref = RAW / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    f = dl.fetcher()
    # corporate actions per quarter, board meetings per month (results dates)
    jobs = []
    q = dt.date(start.year, 1, 1)
    while q <= end:
        q_end = (dt.date(q.year + (q.month + 2) // 12, (q.month + 2) % 12 + 1, 1) - dt.timedelta(days=1))
        jobs.append(("corp_actions", q, min(q_end, end), c["corp_actions_api"]))
        jobs.append(("vix_history", q, min(q_end, end), c["vix_history_api"]))
        q = q_end + dt.timedelta(days=1)
    m = dt.date(start.year, start.month, 1)
    while m <= end:
        m_end = dt.date(m.year + m.month // 12, m.month % 12 + 1, 1) - dt.timedelta(days=1)
        jobs.append(("board_meetings", m, min(m_end, end), c["board_meetings_api"]))
        m = m_end + dt.timedelta(days=1)
    for kind, a, b, tmpl in jobs:
        p = ref / kind / f"{kind}_{a:%Y-%m-%d}_{b:%Y-%m-%d}.json"
        if p.exists():
            continue
        url = tmpl.format(**{"from": f"{a:%d-%m-%Y}", "to": f"{b:%d-%m-%Y}"})
        status, js = f.get_json(url)
        if status == "ok" and isinstance(js, (list, dict)):
            import json
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(js), encoding="utf-8")
        else:
            log(f"  reference {kind} {a}..{b}: {status}")
    for name, url in [("symbolchange.csv", c["symbol_change"])] + \
            [(u.rsplit("/", 1)[-1], u) for u in c["sector_lists"]]:
        p = ref / name
        # never refreshed automatically: the research reads these exact files (reproducibility). Live mode
        # therefore applies symbol changes / sector labels as of 10 Oct 2026 until a manual refresh.
        if p.exists():
            continue
        status, content = f.get(url)
        if status == "ok":
            p.write_bytes(content)
        else:
            log(f"  reference {name}: {status}")
    log(f"reference data done ({f.n_requests} requests so far)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--reference-only", action="store_true")
    ap.add_argument("--no-reference", action="store_true")
    a = ap.parse_args(argv)
    c = config()["data"]
    start = as_date(a.start or c["start"])
    end = min(as_date(a.end or c["end"]), as_date(c["end"]))
    log(f"reused {reuse_delivery_lens_cache()} files from the Delivery Lens cache")
    dl = Downloader()
    if not a.no_reference:
        download_reference(dl, as_date(c["start"]), as_date(c["end"]))
    if a.reference_only:
        return
    dates = candidate_dates(start, end)
    log(f"{len(dates)} candidate dates {start} -> {end}")
    t0, n_trading, n_holiday, n_err = time.time(), 0, 0, 0
    for i, d in enumerate(dates, 1):
        r = dl.fetch_date(d)
        n_trading += r == "trading"
        n_holiday += r == "holiday"
        n_err += r == "error"
        if i % 25 == 0 or i == len(dates):
            el = time.time() - t0
            eta = el / i * (len(dates) - i) / 60
            log(f"PROGRESS {d} | {i}/{len(dates)} dates | trading {n_trading} holiday {n_holiday} "
                f"err {n_err} | {dl.counts} | eta {eta:.0f} min")
        if dl.f is not None and dl.f.consecutive_blocks >= int(c["download"]["max_consecutive_blocks"]):
            log("STOPPED: NSE is blocking repeated requests. Re-run later; cached files are kept.")
            break
    log(f"DONE trading={n_trading} holiday={n_holiday} errors={n_err} counts={dl.counts}")


if __name__ == "__main__":
    main()
