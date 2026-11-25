import datetime as dt
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("should_run", Path(__file__).resolve().parents[1] / "scripts" / "should_run.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)
D = dt.datetime


def test_last_due_slots():
    assert sr.last_due(D(2026, 10, 12, 21, 0)) == D(2026, 10, 12, 19, 30)     # Mon evening -> today
    assert sr.last_due(D(2026, 10, 12, 10, 0)) == D(2026, 10, 9, 19, 30)      # Mon morning -> Friday
    assert sr.last_due(D(2026, 10, 11, 12, 0)) == D(2026, 10, 9, 19, 30)      # Sunday -> Friday


def test_is_due():
    assert sr.is_due(D(2026, 10, 12, 21, 0), None)                            # never ran
    assert sr.is_due(D(2026, 10, 12, 21, 0), D(2026, 10, 9, 20, 10))          # missed Monday 19:30 -> catch up
    assert not sr.is_due(D(2026, 10, 12, 21, 0), D(2026, 10, 12, 20, 15))     # already done tonight
    assert not sr.is_due(D(2026, 10, 13, 9, 0), D(2026, 10, 12, 20, 15))      # next morning: nothing new yet
    assert not sr.is_due(D(2026, 10, 11, 12, 0), D(2026, 10, 9, 20, 0))       # weekend after Friday's run


def test_lock_blocks_a_second_run_and_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "LOCK", tmp_path / "update.lock")
    now = D(2026, 10, 12, 21, 0)
    assert not sr.running(now)
    sr.claim(now)
    assert sr.running(dt.datetime.now())                                       # fresh lock -> another run is skipped
    old = (dt.datetime.now() - dt.timedelta(hours=4)).timestamp()
    import os
    os.utime(sr.LOCK, (old, old))
    assert not sr.running(dt.datetime.now())                                   # a crashed run's lock expires after 3 h
