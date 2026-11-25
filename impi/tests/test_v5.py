"""IMPI v5: strategy grid, filters and exit construction on synthetic data."""
import numpy as np
import pandas as pd

from impi.v5.strategy import BASELINES, select, signals, success_rate, variants


def _events():
    d = pd.Timestamp("2020-01-10")
    return pd.DataFrame({
        "date": [d] * 4, "symbol": ["A", "B", "C", "D"], "family": ["all", "all", "broad", "all"],
        "nifty_above_200": [1.0, 0.0, 1.0, 1.0], "nifty500_ret20": [0.02, -0.01, 0.01, 0.03],
        "dist_52w_high": [-0.05, -0.30, -0.02, -0.08], "rs120": [0.9, 0.95, 0.5, 0.6],
        "entry_date": [d + pd.Timedelta(days=3)] * 4, "entry_px": [100.0] * 4, "liq_bucket": ["large"] * 4,
        "label_10": [1.0, 0.0, 0.0, 1.0], "ret_tb_10": [0.06, -0.03, -0.03, 0.05], "exitk_10": [3.0, 2.0, 1.0, 4.0],
        "fwd_20": [0.10, -0.02, 0.01, 0.0], "exitk_20": [3.0, 2.0, 1.0, 4.0], "ret_tb_20": [0.06, -0.03, -0.03, 0.05],
        "label_20": [1.0, 0.0, 0.0, 1.0]})


def _closes():
    idx = pd.bdate_range("2020-01-01", periods=40)
    return {s: pd.Series(100.0, index=idx) for s in "ABCD"}


def test_grid_is_fixed():
    v = variants()
    assert len(v) == 120 and len(set(v)) == 120
    assert all(b in v for b in BASELINES)


def test_filters():
    ev = _events()
    assert list(select(ev, "B1|M0|S0|TB10")["symbol"]) == ["A", "B", "D"]
    assert list(select(ev, "B1|M1|S0|TB10")["symbol"]) == ["A", "D"]        # Nifty above 200-day average
    assert list(select(ev, "B1|M2|S0|TB10")["symbol"]) == ["A", "D"]        # Nifty 500 up over 20 days
    assert list(select(ev, "B1|M0|S1|TB10")["symbol"]) == ["A", "D"]        # within 10% of 52-week high
    assert list(select(ev, "B1|M0|S2|TB10")["symbol"]) == ["A", "B"]        # top-30% 120-day strength
    assert list(select(ev, "B1|M0|S3|TB10")["symbol"]) == ["A"]
    assert list(select(ev, "BROAD|M0|S0|TB10")["symbol"]) == ["C"]


def test_exit_dates_and_prices():
    ev, cl = _events(), _closes()
    s = signals(select(ev, "B1|M0|S0|TB10"), "B1|M0|S0|TB10", cl)
    idx = cl["A"].index
    t = idx.searchsorted(pd.Timestamp("2020-01-10"))
    assert list(s["exit_date"]) == [idx[t + 3], idx[t + 2], idx[t + 4]]
    assert np.allclose(s["exit_px"], [106.0, 97.0, 105.0])
    s20 = signals(select(ev, "B1|M0|S0|T20"), "B1|M0|S0|T20", cl)
    assert list(s20["exit_date"]) == [idx[t + 20]] * 3
    assert np.allclose(s20["exit_px"], [110.0, 98.0, 100.0])


def test_success_rate():
    ev = _events()
    assert success_rate(select(ev, "B1|M0|S0|TB10"), "B1|M0|S0|TB10") == 2 / 3
    assert success_rate(select(ev, "B1|M0|S0|T20"), "B1|M0|S0|T20") == 1 / 3     # fwd_20 > 0
