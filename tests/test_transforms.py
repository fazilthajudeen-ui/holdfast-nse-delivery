"""Split/bonus adjustment on synthetic data (G6) and the release-lag guard (G1)."""
import numpy as np
import pandas as pd
import pytest

from impi.transforms import adjust
from impi.transforms import pit_store as pit


def _gap_table(rows):
    g = pd.DataFrame(rows, columns=["date", "symbol", "open", "close"])
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values(["symbol", "date"]).reset_index(drop=True)
    g["prev_close_own"] = g.groupby("symbol")["close"].shift(1)
    g["gap"] = g["open"] / g["prev_close_own"]
    g["gap_close"] = g["close"] / g["prev_close_own"]
    g["row"] = g.groupby("symbol").cumcount()
    return g


def _days(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def test_nse_split_confirmed_by_gap_and_crash_not_adjusted():
    d = _days(6)
    rows = []
    # SPLIT: 1:5 split on day 3 (price 500 -> ~101)
    for i, (o, c) in enumerate([(500, 500), (500, 505), (101, 102), (102, 103), (103, 104), (104, 105)]):
        rows.append((d[i], "SPLIT", o, c))
    # CRASH: genuine -45% open with no corporate action -> must NOT be adjusted (no standard ratio)
    for i, (o, c) in enumerate([(100, 100), (100, 100), (55.4, 56), (56, 57), (57, 57), (57, 58)]):
        rows.append((d[i], "CRASH", o, c))
    # HALF: unlisted gap exactly 1:2 -> snapped
    for i, (o, c) in enumerate([(200, 200), (200, 200), (100.5, 101), (101, 101), (101, 102), (102, 102)]):
        rows.append((d[i], "HALF", o, c))
    for k in range(60):                                     # a normal market around them
        rows += [(d[i], f"N{k}", 50.0, 50.0) for i in range(6)]
    g = _gap_table(rows)
    ca = pd.DataFrame({"symbol": ["SPLIT"], "series": ["EQ"], "ex_date": [d[2]],
                       "subject": ["Face Value Split From Rs 10 To Rs 2"], "kind": ["split"], "factor": [0.2]})
    app, logdf = adjust.detect_actions(g, ca)
    a = app.set_index("symbol")
    assert a.loc["SPLIT", "factor"] == pytest.approx(0.2) and a.loc["SPLIT", "method"].startswith("nse_confirmed")
    assert a.loc["SPLIT", "adj_date"] == d[2]
    assert a.loc["HALF", "factor"] == pytest.approx(0.5) and a.loc["HALF", "method"] == "gap_snap"
    assert "CRASH" not in a.index
    assert (logdf["symbol"] == "CRASH").any()


def test_mass_gap_day_is_a_missing_session():
    d = _days(4)
    rows = []
    for k in range(20):                                     # every stock 'halves' on the same day
        rows += [(d[0], f"S{k}", 100, 100), (d[1], f"S{k}", 100, 100), (d[2], f"S{k}", 50, 50), (d[3], f"S{k}", 50, 50)]
    app, logdf = adjust.detect_actions(_gap_table(rows), pd.DataFrame(columns=["symbol", "series", "ex_date", "subject", "kind", "factor"]))
    assert app.empty and (logdf["status"] == "mass_gap_day_not_adjusted").all()


def test_unconfirmed_record_is_not_applied():
    d = _days(4)
    rows = [(d[i], "X", 100, 100) for i in range(4)]       # no gap at all
    g = _gap_table(rows)
    ca = pd.DataFrame({"symbol": ["X"], "series": ["EQ"], "ex_date": [d[2]], "subject": ["Bonus 1:1"],
                       "kind": ["bonus"], "factor": [0.5]})
    app, logdf = adjust.detect_actions(g, ca)
    assert app.empty
    assert logdf.iloc[0]["status"] == "unconfirmed"


def test_cum_factor_prices_and_quantities():
    d = _days(6)
    actions = pd.DataFrame({"symbol": ["S", "S"], "adj_date": [d[2], d[4]], "factor": [0.5, 0.2]})
    suf = adjust.suffix_factors(actions)
    df = pd.DataFrame({"date": d, "symbol": "S"})
    cf = adjust.cum_factor(df, suf)
    np.testing.assert_allclose(cf, [0.1, 0.1, 0.2, 0.2, 1.0, 1.0])
    # a constant market value series becomes continuous after adjustment
    raw = np.array([1000, 1000, 500, 500, 100, 100.0])
    np.testing.assert_allclose(raw * cf, [100] * 6)
    vol = np.array([10, 10, 20, 20, 100, 100.0])
    np.testing.assert_allclose(vol / cf, [100] * 6)


def test_release_lag_guard():
    pit.config.cache_clear()
    for ds in ["equity_bhav", "delivery_mto", "fo_bhav", "participant_oi", "index_close"]:
        pit.assert_usable_at_t(ds)                          # all released before the T decision point
    assert pit.release_offset("ban_list") == pd.Timedelta(days=-1, hours=20)
