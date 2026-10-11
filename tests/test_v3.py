"""IMPI v3: Black-76 implied volatility, per-day option features and NO-LOOK-AHEAD of the option features."""
import numpy as np
import pandas as pd

from impi.v3 import options as O
from impi.v3.panel import option_features


def test_implied_vol_round_trip():
    r = np.random.default_rng(0)
    n = 500
    F = r.uniform(50, 3000, n)
    K = F * np.exp(r.uniform(-0.25, 0.25, n))
    T = r.uniform(3, 90, n) / 365
    sig = r.uniform(0.08, 1.5, n)
    call = r.random(n) < 0.5
    price = O.black76(F, K, T, sig, call)
    iv = O.implied_vol(price, F, K, T, call)
    ok = np.isfinite(iv)
    assert ok.mean() > 0.97                         # only prices below 1e-? tick resolution can fail
    assert np.max(np.abs(iv[ok] - sig[ok])) < 1e-6


def test_implied_vol_rejects_arbitrage_prices():
    F, K, T = np.array([100.0, 100.0]), np.array([90.0, 110.0]), np.array([0.1, 0.1])
    below_intrinsic = np.array([5.0, 0.0])          # call with K=90 worth >= ~10; zero price is not invertible
    assert np.isnan(O.implied_vol(below_intrinsic, F, K, T, np.array([True, True]))).all()


def _synthetic_day(date, F=1000.0, sig=0.3, dte=(20, 48), traded=True):
    rows = []
    for e, d in enumerate(dte):
        exp = date + pd.Timedelta(days=d)
        rows.append({"date": date, "kind": "FUT", "symbol": "ABC", "expiry": exp, "strike": np.nan, "opt": None,
                     "close": F, "contracts": 100})
        for K in np.arange(800, 1201, 20.0):
            for opt in ("CE", "PE"):
                px = float(O.black76(np.array([F]), np.array([K]), np.array([d / 365]), np.array([sig + 0.02 * e]),
                                     np.array([opt == "CE"]))[0])
                rows.append({"date": date, "kind": "OPT", "symbol": "ABC", "expiry": exp, "strike": K, "opt": opt,
                             "close": round(px, 2) if px > 0.05 else 0.05, "contracts": 10 if traded else 0})
    return pd.DataFrame(rows)


def test_day_features_recovers_flat_vol():
    d = pd.Timestamp("2025-03-03")
    f = O.day_features(_synthetic_day(d)).set_index("symbol").loc["ABC"]
    assert abs(f["iv_near"] - 0.30) < 0.005
    assert abs(f["iv_next"] - 0.32) < 0.005
    assert 0.30 < f["iv30"] < 0.32                   # 30-day point lies between the two expiries
    assert abs(f["iv_put_wing"] - 0.30) < 0.01 and abs(f["iv_call_wing"] - 0.30) < 0.01
    assert f["call_contracts"] > 0 and f["fut_contracts"] == 200


def test_untraded_options_are_ignored():
    d = pd.Timestamp("2025-03-03")
    f = O.day_features(_synthetic_day(d, traded=False)).set_index("symbol").loc["ABC"]
    assert np.isnan(f["iv30"]) and f["call_contracts"] == 0


def _iv_history(n=400, seed=2):
    r = np.random.default_rng(seed)
    d = pd.bdate_range("2019-01-01", periods=n)
    iv = 0.3 * np.exp(np.cumsum(r.normal(0, 0.03, n)))
    return pd.DataFrame({"date": d, "symbol": "ABC", "iv_near": iv, "iv_next": iv * 1.02, "iv30": iv * 1.01,
                         "dte_near": 20, "dte_next": 48, "iv_put_wing": iv * 1.1, "iv_call_wing": iv * 1.05,
                         "n_iv_main": 10, "call_contracts": r.integers(1, 1000, n), "put_contracts": r.integers(1, 1000, n),
                         "fut_contracts": r.integers(1, 1000, n)})


def test_option_features_no_look_ahead():
    h = _iv_history()
    full = option_features(h)
    for cut in [150, 277, 399]:
        part = option_features(h.iloc[:cut].copy())
        pd.testing.assert_frame_equal(part.reset_index(drop=True), full.iloc[:cut].reset_index(drop=True),
                                      check_exact=False, rtol=1e-9, atol=1e-12)


def test_forward_tests_detect_skill_and_its_absence():
    from impi.live import forward_tests
    r = np.random.default_rng(5)
    rows = []
    for d in pd.bdate_range("2026-10-12", periods=40):
        n = 150
        y = r.random(n) < 0.3
        good = y + r.normal(0, 0.8, n)                       # informative score
        noise = r.normal(0, 1, n)                            # pure noise
        rows.append(pd.DataFrame({"date": d, "symbol": [f"S{i}" for i in range(n)], "vol_exp": y.astype(float),
                                  "raw3_vol_exp": good, "raw2_vol_exp": noise, "label_xs20": y.astype(float),
                                  "raw3_label_xs20": noise, "label_20": np.nan, "label_xs10": np.nan}))
    t = forward_tests(pd.concat(rows, ignore_index=True))
    assert t["F1"]["supported"] and t["F2"]["supported"]     # skilled v3 beats noisy v2, and beats 0.5
    assert not t["F3"]["supported"]                          # noise score has no skill
    assert "F4" not in t and "F5" not in t                   # nothing matured -> not evaluated
