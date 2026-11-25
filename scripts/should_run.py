"""Is a nightly update due? Exit 0 = yes (run), exit 1 = no (skip).

Used by run_daily.bat so the task can also fire at log-on and on wake from sleep (catch-up after a missed
7:30 PM run) without repeating a run that already succeeded. An update is due when the last SUCCESSFUL run
(data/logs/last_success.txt, written by run_daily.bat) is older than the most recent weekday 19:30 that
has already passed. NSE holidays are not known here; on a holiday the run simply finds nothing new.

    python scripts/should_run.py            # exit code only
    python scripts/should_run.py --force    # always due (manual runs)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

MARKER = Path(__file__).resolve().parents[1] / "data" / "logs" / "last_success.txt"
LOCK = MARKER.with_name("update.lock")          # an update is running (scheduled task or web app); removed at the end
LOCK_MAX_AGE = dt.timedelta(hours=3)             # a crashed run never blocks the next one for longer than this
RUN_AT = dt.time(19, 30)


def last_due(now: dt.datetime) -> dt.datetime:
    d = now.date()
    while True:
        slot = dt.datetime.combine(d, RUN_AT)
        if d.weekday() < 5 and slot <= now:
            return slot
        d -= dt.timedelta(days=1)


def is_due(now: dt.datetime, last_success: dt.datetime | None) -> bool:
    return last_success is None or last_success < last_due(now)


def read_marker() -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(MARKER.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def running(now: dt.datetime) -> bool:
    try:
        return now - dt.datetime.fromtimestamp(LOCK.stat().st_mtime) < LOCK_MAX_AGE
    except OSError:
        return False


def claim(now: dt.datetime) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(now.isoformat(timespec="seconds"), encoding="utf-8")


if __name__ == "__main__":
    now = dt.datetime.now()
    if "--force" in sys.argv:                    # manual run, or the web app that has just claimed the lock itself
        claim(now)
        print(f"[{now:%Y-%m-%d %H:%M}] update forced")
        sys.exit(0)
    if running(now):
        print(f"[{now:%Y-%m-%d %H:%M}] update already running (lock {LOCK.name}) - skip")
        sys.exit(1)
    due = is_due(now, read_marker())
    print(f"[{now:%Y-%m-%d %H:%M}] update due: {due} (last success {read_marker()}, last scheduled slot {last_due(now)})")
    if due:
        claim(now)
    sys.exit(0 if due else 1)
