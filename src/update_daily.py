"""Incremental daily update: download missing days -> rebuild derived data -> log the run.

    python -m src.update_daily            # normal run (safe to run any number of times)
    python -m src.update_daily --force    # rebuild even if no new file arrived
    python -m src.update_daily --push     # also git commit + push the refreshed data
    python -m src.update_daily --loop     # stay running; update Mon-Fri at 19:30 IST

Why rebuild instead of patching rows: a split/bonus on the new day back-adjusts that
stock's entire history, and the rolling baselines/percentiles depend on it. The whole
rebuild takes seconds on 1-2 years of data, so recomputing is simpler and always correct.
If today's file is not published yet (NSE posts it ~6-7 PM IST), the run exits cleanly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time

import pandas as pd

from src import clean, events, features, report, score, stats
from src.common import ROOT, load_config, log, path
from src.download import cached_dates, download_range, download_universe

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
RUN_LOG = "run_history.csv"


def today_ist() -> dt.date:
    return dt.datetime.now(IST).date()


def write_run_log(status: str, new_days: int, last: dt.date | None, secs: float, note: str = ""):
    f = path("logs") / RUN_LOG
    new = not f.exists()
    with open(f, "a", encoding="utf-8") as fh:
        if new:
            fh.write("run_at,status,new_days,data_as_of,seconds,note\n")
        fh.write(f"{dt.datetime.now(IST):%Y-%m-%d %H:%M:%S},{status},{new_days},{last or ''},{secs:.1f},{note}\n")


def rebuild():
    for stage in (clean, features, events, stats, score, report):
        log(f"--- {stage.__name__}")
        stage.main()


def git_push(last: dt.date):
    cmds = [["git", "add", "data/clean", "data/derived", "data/logs/run_history.csv", "reports"],
            ["git", "commit", "-m", f"Data update: NSE data as of {last}"],
            ["git", "push"]]
    for c in cmds:
        r = subprocess.run(c, cwd=ROOT, capture_output=True, text=True)
        log(f"{' '.join(c[:2])}: {'ok' if r.returncode == 0 else r.stderr.strip()[:200]}")
        if r.returncode != 0 and c[1] == "commit" and "nothing to commit" in (r.stdout + r.stderr):
            break


def run(force: bool = False, push: bool = False, source: str | None = None) -> int:
    t0 = time.time()
    cfg = load_config()
    source = source or cfg["source"]
    before = cached_dates()
    today = today_ist()
    if before:
        # look back a week so a day that failed earlier (network error) is retried;
        # cached files and known holidays are skipped without any request
        start = before[-1] - dt.timedelta(days=7)
    else:
        start = (pd.Timestamp(today) - pd.DateOffset(months=cfg["months"])).date()
    new_days = 0
    if start <= today:
        log(f"checking NSE for {start} -> {today}")
        if source == "nse":
            download_universe()
        new_days = download_range(start, today, source)["ok"]
    after = cached_dates()
    last = after[-1] if after else None

    # a previous rebuild may have failed after its files were downloaded (12 Oct 2026): rebuild if the derived
    # scores are older than the newest raw file, even when nothing new was downloaded this time
    stale = False
    try:
        built = pd.read_parquet(path("scores"), columns=["date"])["date"].max()
        stale = last is not None and pd.Timestamp(built).date() < last
    except Exception:
        stale = last is not None
    if stale and new_days == 0:
        log(f"derived scores are older than the newest raw file ({last}) - rebuilding")
    if new_days == 0 and not force and not stale:
        msg = (f"No new file. Data is already as of {last}. "
               + ("Today's bhavcopy is usually published ~6-7 PM IST." if last != today else "Up to date."))
        log(msg)
        write_run_log("no_new_data", 0, last, time.time() - t0)
        return 0
    try:
        rebuild()
    except Exception as e:
        write_run_log("error", new_days, last, time.time() - t0, repr(e)[:150].replace(",", ";"))
        raise
    write_run_log("updated", new_days, last, time.time() - t0)
    log(f"update complete: {new_days} new day(s), data as of {last}, {time.time() - t0:.0f}s")
    if push or cfg.get("git_push"):
        git_push(last)
    return new_days


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--loop", action="store_true", help="keep running; update Mon-Fri 19:30 IST")
    ap.add_argument("--source", choices=["nse", "github"])
    a = ap.parse_args(argv)
    if not a.loop:
        run(a.force, a.push, a.source)
        return

    import schedule
    # schedule uses the laptop's local clock; convert 19:30 IST to local time
    local_offset = dt.datetime.now().astimezone().utcoffset()
    at = (dt.datetime(2000, 1, 1, 19, 30) - dt.timedelta(hours=5, minutes=30) + local_offset).strftime("%H:%M")
    job = lambda: run(push=a.push, source=a.source)
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        getattr(schedule.every(), day).at(at).do(job)
    log(f"scheduler running: Mon-Fri at {at} local time (19:30 IST). Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
