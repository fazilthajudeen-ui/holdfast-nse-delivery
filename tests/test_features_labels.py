"""Features, VCP detector, labels and events on synthetic data, including NO-LOOK-AHEAD checks (G1)."""
import numpy as np
import pandas as pd
import pytest

from impi.common import config
from impi.features.compression import compression_features
from impi.features.delivery import delivery_features
from impi.features.positioning import cot_index, market_positioning, run_length, stock_positioning
from impi.features.vcp import VCPDetector, fractal_pivots
from impi.features.volume import volume_features
from impi.labels.events import apply_cooldown, compression_regime, earnings_mask
from impi.labels.triple_barrier import triple_barrier
from impi.labels.vol_expansion import vol_expansion

FCFG = config()["features"]


def synthetic_stock(n=600, seed=1):
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.015, n)))
    openp = close * np.exp(r.normal(0, 0.004, n))
    high = np.maximum(openp, close) * (1 + np.abs(r.normal(0, 0.006, n)))
    low = np.minimum(openp, close) * (1 - np.abs(r.normal(0, 0.006, n)))
    vol = r.lognormal(12, 0.4, n)
    oi = np.abs(1e6 + np.cumsum(r.normal(0, 2e4, n)))
    d = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"date": d, "open": openp, "high": high, "low": low, "close": close,
                         "raw_close": close, "volume": vol, "deliv_pct": r.uniform(20, 70, n),
                         "turnover_rs": vol * close, "fut_oi_adj": oi, "call_oi": oi * 0.5, "put_oi": oi * 0.4,
                         "near_settle": close * 1.003, "near_expiry": d + pd.Timedelta(days=20),
                         "next_settle": close * 1.008, "next_expiry": d + pd.Timedelta(days=50)})


def all_features(df):
    return pd.concat([delivery_features(df, FCFG), volume_features(df, FCFG),
                      compression_features(df, FCFG), stock_positioning(df, FCFG)], axis=1)


def test_no_look_ahead_truncation_invariance():
    df = synthetic_stock()
    full = all_features(df)
    for cut in [300, 451, 599]:
        part = all_features(df.iloc[:cut].copy())
        pd.testing.assert_frame_equal(part, full.iloc[:cut], check_exact=False, rtol=1e-9, atol=1e-12)


def test_vcp_no_look_ahead():
    df = synthetic_stock(n=500, seed=3)
    full = VCPDetector(df.high, df.low, df.close, df.volume, FCFG["vcp"])
    for cut in [200, 333, 480]:
        part = VCPDetector(df.high[:cut], df.low[:cut], df.close[:cut], df.volume[:cut], FCFG["vcp"])
        for t in range(cut - 30, cut):
            assert part.pattern_at(t) == full.pattern_at(t)


def test_feature_ranges():
    f = all_features(synthetic_stock())
    assert f["bbw_pct252"].dropna().between(0, 100).all()
    assert f["oi_cot"].dropna().between(0, 100).all()
    assert (f["atr_ratio"].dropna() > 0).all()
    assert f["nr7_count10"].dropna().between(0, 10).all()
    q = f[["long_buildup", "short_covering", "short_buildup", "long_unwinding"]].dropna()
    assert (q.sum(axis=1) <= 1).all()


def test_cot_and_run_length():
    x = pd.Series([1, 2, 3, 2, 1, 1, 5.0])
    np.testing.assert_allclose(cot_index(x, 3, 3).to_numpy()[2:], [100, 0, 0, 0, 100])
    np.testing.assert_allclose(run_length(x).to_numpy(), [0, 1, 2, -1, -2, 0, 1])


def test_market_positioning_share_is_lot_size_invariant():
    d = pd.bdate_range("2020-01-01", periods=300)
    rows = []
    for i, dt in enumerate(d):
        lot = 3 if i >= 150 else 1                         # contract counts drop 3x after a lot revision
        fl, fs = (500 + i) / lot, 400 / lot
        tot = 2000 / lot
        rows.append({"date": dt, "participant": "FII", "fut_idx_long": fl, "fut_idx_short": fs,
                     "fut_stk_long": fl, "fut_stk_short": fs})
        rows.append({"date": dt, "participant": "TOTAL", "fut_idx_long": tot, "fut_idx_short": tot,
                     "fut_stk_long": tot, "fut_stk_short": tot})
        for p in ["DII", "Pro", "Client"]:
            rows.append({"date": dt, "participant": p, "fut_idx_long": 1, "fut_idx_short": 1,
                         "fut_stk_long": 1, "fut_stk_short": 1})
    m = market_positioning(pd.DataFrame(rows), FCFG).set_index("date")
    share = m["fii_fut_idx_share"]
    assert share.diff().iloc[150] == pytest.approx(share.diff().iloc[149], rel=1e-9)   # no jump at the revision


def _vcp_series():
    """Hand-built VCP: base high at bar 300, three contractions 21% / 11% / 5%, falling volume,
    breakout at bar 350 on 4x volume."""
    knots = [(0, 50), (300, 100), (310, 80), (320, 96), (328, 86.4), (336, 93), (341, 89), (349, 92.5), (350, 95), (360, 96)]
    close = np.interp(np.arange(361), [k for k, _ in knots], [v for _, v in knots])
    vol = np.full(361, 1200.0)
    for a, b, v in [(300, 310, 1000), (310, 320, 900), (320, 328, 700), (328, 336, 600), (336, 341, 400), (341, 350, 500)]:
        vol[a:b] = v
    vol[350] = 2000
    vol[351:] = 800
    return close * 1.005, close * 0.995, close, vol


def test_vcp_detects_textbook_pattern():
    h, l, c, v = _vcp_series()
    det = VCPDetector(h, l, c, v, FCFG["vcp"])
    p = det.pattern_at(350)
    assert p["vcp_valid"] == 1.0 and p["vcp_n"] == 3
    assert p["vcp_depth_first"] > p["vcp_depth_last"]
    assert p["vcp_pivot"] == pytest.approx(93 * 1.005)
    vr = v / pd.Series(v).shift(1).rolling(20).mean().to_numpy()
    trig = [t for t, _ in det.triggers(vr)]
    assert trig == [350]


def test_vcp_rejects_widening_contractions_and_rising_volume():
    h, l, c, v = _vcp_series()
    c2 = c.copy()
    c2[336:342] = np.interp(np.arange(336, 342), [336, 341], [93, 70])       # third contraction deeper
    det = VCPDetector(c2 * 1.005, c2 * 0.995, c2, v, FCFG["vcp"])
    assert det.pattern_at(350)["vcp_valid"] == 0.0
    v2 = v.copy()
    v2[336:341] = 5000                                                    # volume rising in last contraction
    det = VCPDetector(h, l, c, v2, FCFG["vcp"])
    assert det.pattern_at(350)["vcp_valid"] == 0.0


def test_fractal_pivots_need_k_bars_each_side():
    high = np.array([1, 2, 5, 2, 1, 3, 9.0])
    ph, _ = fractal_pivots(high, high, 2)
    assert ph.tolist() == [False, False, True, False, False, False, False]   # last bars unconfirmed


def test_triple_barrier_cases():
    # entry at open of bar 1 = 100, ATR 2 -> upper 104, lower 98
    o = np.array([100, 100, 101, 102, 103, 100, 100, 100, 100, 100, 100, 100.0])
    h = o + 0.5
    l = o - 0.5
    c = o.copy()
    atr = np.full(len(o), 2.0)
    h[4] = 104.2                                                     # upper touched on bar 4
    r = triple_barrier(o, h, l, c, atr, [0], 10, 2.0, 1.0, False)
    assert r["label"][0] == 1 and r["exit_k"][0] == 4 and r["ret"][0] == pytest.approx(0.04)
    l2 = l.copy()
    l2[2] = 97.0                                                     # lower first
    r = triple_barrier(o, h, l2, c, atr, [0], 10, 2.0, 1.0, False)
    assert r["label"][0] == 0 and r["exit_k"][0] == 2 and r["ret"][0] == pytest.approx(-0.02)
    h3, l3 = h.copy(), l.copy()
    h3[3], l3[3] = 105, 97                                           # both same bar -> loss
    r = triple_barrier(o, h3, l3, c, atr, [0], 10, 2.0, 1.0, False)
    assert r["label"][0] == 0 and r["exit_k"][0] == 3
    o4 = o.copy()
    o4[3] = 96.0                                                     # gap below the barrier -> fill at open
    l4 = l.copy()
    l4[3] = 95.5
    r = triple_barrier(o4, h, l4, c, atr, [0], 10, 2.0, 1.0, False)
    assert r["ret"][0] == pytest.approx(-0.04)
    flat_h, flat_l = o + 0.1, o - 0.1                                # nothing touched -> time barrier
    r = triple_barrier(o, flat_h, flat_l, c, atr, [0], 10, 2.0, 1.0, False)
    assert r["label"][0] == 0 and r["exit_k"][0] == 10
    r = triple_barrier(o, flat_h, flat_l, c, atr, [5], 10, 2.0, 1.0, False)  # data ends -> incomplete
    assert np.isnan(r["label"][0])
    r = triple_barrier(o, flat_h, flat_l, c, atr, [5], 10, 2.0, 1.0, True)   # delisted -> loss at last close
    assert r["label"][0] == 0 and r["exit_k"][0] == 6


def test_vol_expansion_label():
    c = np.r_[100 * np.exp(np.cumsum(np.tile([0.01, -0.01], 10))), 100 * np.exp(np.cumsum(np.tile([0.05, -0.05], 10)))]
    lab, ratio = vol_expansion(c, [19], 10, 10, 1.5)
    assert lab[0] == 1 and ratio[0] > 4


def test_cooldown_and_compression_regime():
    assert apply_cooldown(np.array([5, 6, 15, 16, 30]), 10).tolist() == [5, 16, 30]
    bbw = pd.Series([10.0] * 5 + [50.0] * 5 + [10.0])
    reg = compression_regime(bbw, 30, 5, 10)
    assert reg[10] and not reg[9]


def test_earnings_mask_uses_only_known_future_dates():
    gidx = np.arange(20)
    dates = pd.bdate_range("2020-01-01", periods=20).to_numpy()
    known_late = pd.Timestamp(dates[9]) + pd.Timedelta(hours=10)     # intimated on day 9 morning
    m = earnings_mask(gidx, dates, np.array([10]), np.array([known_late]), 3, pd.Timedelta(hours=23, minutes=59))
    # days 7-8 are within 3 days before the results but the meeting was not yet announced -> kept;
    # day 9 (announced that morning) .. day 13 (3 days after) -> excluded; day 14 -> kept
    assert m.tolist()[6:15] == [False, False, False, True, True, True, True, True, False]
    known_early = pd.Timestamp(dates[2])
    m2 = earnings_mask(gidx, dates, np.array([10]), np.array([known_early]), 3, pd.Timedelta(hours=23, minutes=59))
    assert m2[7] and m2[13] and not m2[6]
