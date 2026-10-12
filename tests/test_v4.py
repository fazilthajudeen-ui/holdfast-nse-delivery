"""IMPI v4: signed delivery (no look-ahead) and the pre-registered statistics on synthetic data."""
import numpy as np
import pandas as pd

from impi.v4.signals import signed_delivery
from impi.v4.stats import c1_test, family, ic_test, rate_test


def _stock(n=300, seed=1):
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.01, n)))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "close": close,
                         "deliv_pct": r.uniform(20, 70, n)})


def test_signed_delivery_no_look_ahead():
    df = _stock()
    full = signed_delivery(df)
    for cut in [120, 211, 299]:
        part = signed_delivery(df.iloc[:cut].copy())
        pd.testing.assert_series_equal(part, full.iloc[:cut], check_exact=False, rtol=1e-12)


def test_signed_delivery_sign():
    # constant high delivery on rising days -> positive; on falling days -> negative
    n = 120
    up = pd.DataFrame({"close": np.linspace(100, 200, n), "deliv_pct": np.r_[np.full(n - 10, 30.0), np.full(10, 60.0)]})
    dn = pd.DataFrame({"close": np.linspace(200, 100, n), "deliv_pct": up["deliv_pct"]})
    assert signed_delivery(up).iloc[-1] > 0 and signed_delivery(dn).iloc[-1] < 0


def _panel(signal_strength, n_dates=120, n_stocks=60, seed=3):
    r = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2023-01-02", periods=n_dates):
        s = r.normal(size=n_stocks)
        y = signal_strength * s + r.normal(size=n_stocks)
        rows.append(pd.DataFrame({"date": d, "sig": s, "out": y}))
    return pd.concat(rows, ignore_index=True)


def test_ic_test_detects_signal_and_noise():
    good = ic_test(_panel(0.3), "D1", "sig", "out")
    none = ic_test(_panel(0.0), "D1", "sig", "out")
    assert good["estimate"] > 0.15 and good["p"] < 0.01
    assert none["p"] > 0.05


def test_rate_test_and_min_group():
    r = np.random.default_rng(4)
    n = 2000
    exp = r.random(n) < 0.5
    y = (r.random(n) < np.where(exp, 0.40, 0.25)).astype(float)
    ev = pd.DataFrame({"date": pd.Timestamp("2023-01-02") + pd.to_timedelta(r.integers(0, 900, n), "D"), "e": exp, "y": y})
    out = rate_test(ev, "V1", "e", "y")
    assert out["estimate"] > 0.1 and out["p"] < 1e-6
    small = rate_test(ev.iloc[:40], "V1", "e", "y")
    assert np.isnan(small["p"]) and "insufficient" in small["label"]


def test_c1_and_family():
    r = np.random.default_rng(6)
    n = 900
    on = np.zeros(n, bool)
    on[100:160] = on[500:560] = True
    y = r.normal(0.02, 0.03, n) + np.where(on, 0.05, 0.0)
    m = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=n), "fii_fut_stk_cot": np.where(on, 5.0, 50.0),
                      "c1_on": on, "n500_fwd60": y})
    c1 = c1_test(m)
    assert c1["estimate"] > 0.03
    res = family([{"hypothesis": "D1", "estimate": 0.1, "p": 0.001}, {"hypothesis": "H2p", "estimate": 0.02, "p": 0.001},
                  {"hypothesis": "V1", "estimate": 0.1, "p": np.nan}])
    assert res[0]["supported"] and not res[1]["supported"]          # H2p expects a negative difference
    assert not res[2]["supported"] and res[2]["bh_q"] != res[2]["bh_q"]
