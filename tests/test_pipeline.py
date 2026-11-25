import numpy as np
import pandas as pd
import pytest

from src.clean import adjust_corporate_actions, parse_bhav, pick_chain_rows
from src.common import load_config
from src.download import file_date, is_valid_bhav, is_valid_index
from src.events import Market, assign_groups, build_events, detect_moves
from src.features import add_features
from src.score import rolling_percentile

SAMPLE = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE,"
    " AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "ABC,  EQ, 25-Sep-2026, 100.00, 101.00, 105.00, 99.50, 104.00, 104.50, 102.3, 1000, 1.02, 50, 400, 40.00\n"
    "XYZ,  BL, 25-Sep-2026, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 50.00, 200, 0.10, 1, -, -\n"
)


# ---------------------------------------------------------------- parsing / download
def test_parse_strips_spaces_and_dashes():
    df = parse_bhav(SAMPLE)
    assert list(df.columns)[:3] == ["symbol", "series", "date"]
    assert df.loc[0, "series"] == "EQ"                      # leading spaces stripped
    assert df.loc[0, "date"] == pd.Timestamp("2026-09-25")
    assert df.loc[0, "close"] == 104.5 and df.loc[0, "deliv_qty"] == 400
    assert np.isnan(df.loc[1, "deliv_qty"]) and np.isnan(df.loc[1, "deliv_per_nse"])


def test_holiday_html_detected():
    html = "<!DOCTYPE html>\n<html lang='en'><head><title>404</title></head></html>"
    assert not is_valid_bhav(html)
    assert is_valid_bhav(SAMPLE)
    assert is_valid_bhav("﻿" + SAMPLE)                 # BOM tolerated
    assert is_valid_index("Index Name,Index Date,Open Index Value")
    assert not is_valid_index(html)


def test_stale_holiday_file_detected_by_internal_date():
    # NSE served 30-Apr data for the 1-May holiday: the header is valid, the date is not
    import datetime as dt
    assert file_date(SAMPLE) == dt.date(2026, 9, 25)
    assert file_date(SAMPLE) != dt.date(2026, 9, 26)
    assert file_date("Index Name,Index Date\nNifty 50,25-09-2026,1", "index") == dt.date(2026, 9, 25)
    assert file_date("<html></html>") is None


# ---------------------------------------------------------------- corporate actions
def _chain(closes, prev_closes, vols, symbol="ABC", series="EQ"):
    d = pd.bdate_range("2026-01-01", periods=len(closes))
    return pd.DataFrame({"symbol": symbol, "series": series, "date": d, "close": closes,
                         "prev_close": prev_closes, "open": closes, "high": closes, "low": closes,
                         "last": closes, "avg_price": closes, "volume": vols, "deliv_qty": np.array(vols) / 2})


def test_split_back_adjusts_prices_and_quantities():
    # 1:2 split on day 3: price 200 -> ~100; NSE sets PREV_CLOSE = 100 (already adjusted)
    df = _chain([200, 202, 204, 103, 104], [198, 200, 202, 102, 103], [1000, 1000, 1000, 2000, 2000])
    adj, actions = adjust_corporate_actions(df, 0.02)
    assert len(actions) == 1
    assert actions.iloc[0]["factor"] == pytest.approx(0.5)
    assert actions.iloc[0]["implied_ratio"] == pytest.approx(2.0)
    assert adj["close"].tolist() == pytest.approx([100, 101, 102, 103, 104])
    assert adj["volume"].tolist() == pytest.approx([2000, 2000, 2000, 2000, 2000])
    assert adj["deliv_qty"].tolist() == pytest.approx([1000] * 5)


def test_split_with_unadjusted_prev_close_detected_from_open_gap():
    # real NSE behaviour: ex-date PREV_CLOSE is the raw close; open is ~1/5 of it (1:5 split)
    df = _chain([700, 709.4, 170.25, 172], [690, 700, 709.4, 170.25], [100, 100, 900, 800])
    df.loc[2, "open"] = 148.2                                # +4.5% off the exact 1/5 ratio
    adj, actions = adjust_corporate_actions(df, 0.02)
    assert len(actions) == 1 and actions.iloc[0]["method"] == "open_gap_snapped"
    assert actions.iloc[0]["factor"] == pytest.approx(0.2)
    assert adj["close"].iloc[:2].tolist() == pytest.approx([140, 141.88])
    assert adj["volume"].iloc[0] == pytest.approx(500)


def test_real_crash_not_mistaken_for_split():
    df = _chain([100, 100, 60, 58], [100, 100, 100, 60], [10] * 4)
    df.loc[2, "open"] = 57                                   # -43% gap, not near a standard ratio
    _, actions = adjust_corporate_actions(df, 0.02)
    assert actions.empty


def test_no_false_corporate_action_on_normal_days():
    df = _chain([100, 101, 99, 100.5], [99, 100, 101, 99], [10] * 4)
    _, actions = adjust_corporate_actions(df, 0.02)
    assert actions.empty


def test_mass_prev_close_jump_is_missing_session_not_split():
    # 60 stocks all show a 3% PREV_CLOSE jump on the same day (a skipped Sunday session)
    frames = [_chain([100, 101, 104, 105], [99, 100, 104, 104], [10] * 4, symbol=f"S{i}") for i in range(60)]
    df = pd.concat(frames, ignore_index=True)
    _, flagged = adjust_corporate_actions(df, 0.02)
    assert len(flagged) == 60                               # naive rule: 60 fake "splits"
    adj, actions = adjust_corporate_actions(df, 0.02, mass_event_share=0.05)
    assert actions.empty and (adj["adj_mult"] == 1).all()


def test_series_hop_uses_chain_not_false_split():
    # stock trades EQ, moves to BE for a day (price falls 10%), back to EQ
    eq = _chain([100, 101, 91, 92], [99, 100, 101, 91], [10] * 4)
    eq.loc[2, "series"] = "BE"
    raw = pd.concat([eq, eq.iloc[[2]].assign(series="BL")])   # a duplicate block-deal row
    chain = pick_chain_rows(raw, ["EQ", "BE", "BZ"])
    assert len(chain) == 4 and chain.iloc[2]["series"] == "BE"
    _, actions = adjust_corporate_actions(chain, 0.02)
    assert actions.empty


# ---------------------------------------------------------------- events
def test_detect_up_move_and_cooldown():
    c = np.full(120, 100.0)
    c[30:40] = np.linspace(100, 125, 10)       # +25% over 9 days -> event when >= +20%
    c[40:] = 125
    c[80:86] = np.linspace(125, 160, 6)        # second rally, outside cooldown
    ev = detect_moves(c, 0.20, 20, 40, "up")
    assert len(ev) == 2
    t, m = ev[0]
    assert c[t] / c[m] - 1 >= 0.20 and c[t - 1] / c[m] - 1 < 0.20   # first day it crosses
    assert ev[1][0] - t > 40
    # a rally inside the cool-down is ignored
    c2 = c.copy()
    c2[55:60] = np.linspace(125, 155, 5)
    c2[60:] = 155
    ev2 = detect_moves(c2, 0.20, 20, 40, "up")
    assert all(b[0] - a[0] > 40 for a, b in zip(ev2, ev2[1:]))


def test_detect_down_move():
    c = np.r_[np.full(30, 100.0), np.linspace(100, 70, 11), np.full(30, 70.0)]
    ev = detect_moves(c, 0.20, 20, 40, "down")
    assert len(ev) == 1
    t, m = ev[0]
    assert c[t] / c[m] - 1 <= -0.20


def test_event_outcomes_on_synthetic_market():
    n = 200
    dates = pd.bdate_range("2025-01-01", periods=n)
    c = np.full(n, 100.0)
    c[100:106] = np.linspace(100, 130, 6)      # +30% rally, event on the day it crosses +20%
    c[106:] = 130
    c[130:] = 110                               # gives back 20 of 30 points -> round-trip
    vol = np.full(n, 1000.0)
    deliv = np.full(n, 300.0)
    deliv[101:106] = 800                        # heavy delivery during the move
    mkt = Market(dates=dates, symbols=["ABC"], close=c[:, None], volume=vol[:, None],
                 deliv=deliv[:, None], baseline=np.full((n, 1), 30.0),
                 bench_fwd={h: np.zeros(n) for h in (5, 20, 60)}, bench_name="flat",
                 analysis_start=dates[0])
    cfg = load_config()
    ev = build_events(mkt, 0.20, 20, 40, directions=("up",), cfg=cfg)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["event_deliv"] == pytest.approx(80.0)
    assert e["event_rel_deliv"] == pytest.approx(50.0)
    assert e["round_trip"] == 1.0
    assert e["status"] == "complete" and e["complete_60"]
    assert e["exc_60"] == pytest.approx(e["ret_60"])      # flat benchmark
    assert e["max_dd_60"] < 0


def test_incomplete_events_are_flagged():
    n = 60
    dates = pd.bdate_range("2025-01-01", periods=n)
    c = np.r_[np.full(40, 100.0), np.linspace(100, 125, 5), np.full(15, 125.0)]
    mkt = Market(dates=dates, symbols=["ABC"], close=c[:, None], volume=np.ones((n, 1)),
                 deliv=np.ones((n, 1)), baseline=np.full((n, 1), 50.0),
                 bench_fwd={h: np.zeros(n) for h in (5, 20, 60)}, bench_name="flat",
                 analysis_start=dates[0])
    e = build_events(mkt, 0.20, 20, 40, directions=("up",)).iloc[0]
    assert e["status"] == "incomplete"
    assert e["complete_5"] and not e["complete_60"]
    assert np.isnan(e["ret_60"]) and np.isnan(e["round_trip"])


def test_groups_are_terciles():
    g = assign_groups(pd.Series(np.arange(9.0)), 3)
    assert g.tolist() == ["Low"] * 3 + ["Mid"] * 3 + ["High"] * 3


# ---------------------------------------------------------------- no look-ahead
def _synthetic_panel(n_days=150, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2025-01-01", periods=n_days)
    vol = rng.integers(1000, 5000, n_days).astype(float)
    return pd.DataFrame({"symbol": "ABC", "date": d,
                         "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))),
                         "volume": vol, "deliv_qty": vol * rng.uniform(0.2, 0.7, n_days)})


def test_baseline_does_not_change_when_future_rows_added():
    full = add_features(_synthetic_panel(150))
    cut = add_features(_synthetic_panel(150).iloc[:100])
    cols = ["baseline_60", "rel_deliv", "deliv_qty_ratio", "rel_deliv_5", "ret_20"]
    pd.testing.assert_frame_equal(full[cols].iloc[:100].reset_index(drop=True),
                                  cut[cols].reset_index(drop=True))


def test_baseline_excludes_today():
    df = _synthetic_panel(100)
    df.loc[99, "deliv_qty"] = df.loc[99, "volume"]          # 100% delivery today
    f = add_features(df)
    past = f["deliv_pct"].iloc[39:99].median()
    assert f["baseline_60"].iloc[99] == pytest.approx(past)


def test_regime_hysteresis_and_zones():
    from src.regime import regimes, zones
    s = np.array([np.nan, 0, 3, 6, 4, 2.5, 1, 0, -6, -3, -1, 0.0])
    st = regimes(s, 5, 2)
    # enters at 6, stays through 4 and 2.5 (still >= 2), leaves at 1; low zone from -6, leaves at -1
    assert st.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, -1, -1, 0, 0]
    z = zones(pd.Series(pd.bdate_range("2026-01-01", periods=len(s))), st)
    assert [(x["kind"], x["days"]) for x in z] == [("accumulation", 3), ("low", 2)]


def test_regime_is_causal():
    from src.regime import regimes, smooth_rel
    rel = pd.Series(np.random.default_rng(3).normal(0, 6, 200))
    full = regimes(smooth_rel(rel, 10).to_numpy(), 5, 2)
    part = regimes(smooth_rel(rel.iloc[:120], 10).to_numpy(), 5, 2)
    assert (full[:120] == part).all()


def test_score_is_causal_and_bounded():
    x = np.random.default_rng(1).normal(size=300)
    full = rolling_percentile(x, 250, 60)
    part = rolling_percentile(x[:200], 250, 60)
    np.testing.assert_allclose(full[:200], part, equal_nan=True)
    assert np.nanmin(full) >= 0 and np.nanmax(full) <= 100
    assert np.isnan(full[:59]).all() and not np.isnan(full[59])
    # the maximum of its own history scores near 100
    y = np.arange(100.0)
    assert rolling_percentile(y, 250, 60)[-1] == pytest.approx(99.5)
