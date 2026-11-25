"""Statistics, validation, costs and backtest engine - checked against independent implementations."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from impi.backtest import costs as C
from impi.backtest.executor import run_backtest
from impi.backtest.metrics import deflated_sharpe, max_drawdown, portfolio_metrics, spa_tests
from impi.models.composite import Composite
from impi.research.stats import delong_test, mde_two_proportions, month_block_bootstrap, two_proportion_z
from impi.validation.multiple_testing import benjamini_hochberg
from impi.validation.splits import assign_split, purged_kfold


def test_bh_matches_statsmodels():
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.2, 0.5, np.nan])
    q, rej = benjamini_hochberg(p, 0.10)
    rej_sm, q_sm, _, _ = multipletests(p[:-1], alpha=0.10, method="fdr_bh")
    np.testing.assert_allclose(q[:-1], q_sm)
    assert rej[:-1].tolist() == rej_sm.tolist() and not rej[-1] and np.isnan(q[-1])


def test_two_proportion_matches_statsmodels():
    r = two_proportion_z(60, 150, 40, 140)
    z, p = proportions_ztest([60, 40], [150, 140])
    assert r["z"] == pytest.approx(z) and r["p"] == pytest.approx(p)


def test_delong_auc_and_null_behaviour():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    a = y * 0.8 + rng.normal(0, 1, 2000)
    b = y * 0.3 + rng.normal(0, 1, 2000)
    r = delong_test(y, a, b)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, a)) and r["auc_b"] == pytest.approx(roc_auc_score(y, b))
    assert r["p"] < 1e-6
    # identical-quality scores -> p-values roughly uniform; check the SE against a bootstrap
    c = y * 0.5 + rng.normal(0, 1, 2000)
    d = y * 0.5 + rng.normal(0, 1, 2000)
    r2 = delong_test(y, c, d)
    diffs = []
    for _ in range(300):
        i = rng.integers(0, 2000, 2000)
        diffs.append(roc_auc_score(y[i], c[i]) - roc_auc_score(y[i], d[i]))
    assert r2["se"] == pytest.approx(np.std(diffs), rel=0.2)


def test_month_block_bootstrap_reproducible():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=400), "x": rng.normal(0.1, 1, 400)})
    f = lambda d: float(d["x"].mean())
    a = month_block_bootstrap(df, f, 200, np.random.default_rng(5))
    b = month_block_bootstrap(df, f, 200, np.random.default_rng(5))
    assert a == b and a["ci_low"] < a["estimate"] < a["ci_high"]


def test_mde_shrinks_with_n():
    assert mde_two_proportions(0.4, 1000, 1000) < mde_two_proportions(0.4, 100, 100)


def test_assign_split_purges_boundary_labels():
    dates = pd.Series(pd.to_datetime(["2020-12-20", "2020-12-01", "2021-03-01", "2023-05-01"]))
    le = pd.Series(pd.to_datetime(["2021-01-05", "2020-12-15", "2021-03-15", "2023-05-15"]))
    s = assign_split(dates, le)
    assert pd.isna(s.iloc[0]) and s.iloc[1:].tolist() == ["train", "validation", "test"]


def test_purged_kfold_no_overlap_and_embargo():
    cal = pd.bdate_range("2020-01-01", periods=400).to_numpy()
    dates = pd.Series(cal[:380])
    le = pd.Series(cal[10:390])                            # each label spans 10 trading days
    for tr, te in purged_kfold(dates, le, 5, 10, cal):
        a, b = dates.iloc[te].min(), dates.iloc[te].max()
        dtr, ltr = dates.iloc[tr], le.iloc[tr]
        assert not ((dtr <= b) & (ltr >= a)).any()          # no training label overlaps the test window
        bi = np.searchsorted(cal, np.datetime64(b))
        assert not ((dtr > b) & (dtr <= cal[min(bi + 10, len(cal) - 1)])).any()


def test_costs_round_trip_reasonable():
    rt = C.round_trip_pct(50000, "large")
    # STT 0.2% + stamp 0.015% + 2x Rs20 brokerage (0.08%) + DP + exchange/GST + 10 bps slippage ~ 0.43%
    assert 0.0035 < rt < 0.0050
    assert C.round_trip_pct(50000, "small") > rt


def test_backtest_single_trade_accounting():
    cal = pd.bdate_range("2021-01-01", periods=10)
    sig = pd.DataFrame({"date": [cal[0]], "symbol": ["A"], "entry_date": [cal[1]], "exit_date": [cal[4]],
                        "entry_px": [100.0], "exit_px": [110.0], "liq_bucket": ["large"], "family": ["x"]})
    closes = {"A": pd.Series([100, 100, 104, 106, 110, 110, 110, 110, 110, 110.0], index=cal)}
    ret, tr = run_backtest(sig, closes, cal.to_numpy(), seed=1, max_positions=20, capital=1_000_000, cash_yield=0.0)
    assert len(tr) == 1
    notional = 1_000_000 / 20
    gross = notional * (110 * (1 - 5e-4)) / (100 * (1 + 5e-4))
    expected = gross - C.sell_cost(gross) - (notional + C.buy_cost(notional))
    assert tr.iloc[0]["pnl"] == pytest.approx(expected, rel=1e-9)
    equity_end = 1_000_000 * (1 + ret).prod()
    assert equity_end == pytest.approx(1_000_000 + expected, rel=1e-9)


def test_backtest_respects_position_cap():
    cal = pd.bdate_range("2021-01-01", periods=10)
    n = 30
    sig = pd.DataFrame({"date": cal[0], "symbol": [f"S{i}" for i in range(n)], "entry_date": cal[1],
                        "exit_date": cal[5], "entry_px": 100.0, "exit_px": 100.0, "liq_bucket": "large",
                        "priority": np.arange(n, dtype=float)})
    closes = {f"S{i}": pd.Series(100.0, index=cal) for i in range(n)}
    _, tr = run_backtest(sig, closes, cal.to_numpy(), seed=1, max_positions=20)
    assert len(tr) == 20 and set(tr["symbol"]) == {f"S{i}" for i in range(10, 30)}   # highest priority wins


def test_metrics_and_drawdown():
    eq = pd.Series([100, 120, 90, 130.0])
    assert max_drawdown(eq) == pytest.approx(-0.25)
    r = pd.Series(np.random.default_rng(3).normal(0.001, 0.01, 1000))
    m = portfolio_metrics(r, 0.0)
    assert m["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(252))


def test_deflated_sharpe_penalises_many_trials():
    few = deflated_sharpe(0.08, 1000, 0.0, 3.0, 0.0004, 2)["dsr"]
    many = deflated_sharpe(0.08, 1000, 0.0, 3.0, 0.0004, 500)["dsr"]
    assert many < few


def test_spa_runs_and_detects_obvious_winner():
    rng = np.random.default_rng(4)
    idx = pd.RangeIndex(1000)
    bench = pd.Series(rng.normal(0, 0.01, 1000), idx)
    variants = pd.DataFrame({"good": bench + 0.003, "noise": rng.normal(0, 0.01, 1000)}, idx)
    r = spa_tests(bench, variants, reps=300, block=10, seed=1)
    assert r["spa_p_consistent"] < 0.05 and r["white_rc_p"] < 0.05


def test_composite_uses_train_statistics_only():
    rng = np.random.default_rng(2)
    def frame(n, shift):
        return pd.DataFrame({c: rng.normal(shift, 1, n) for c in
                             ["bbw_pct252", "atr_ratio", "nr7_count10", "range_contraction", "deliv_z20",
                              "price_oi_align", "vol_z"]}).assign(fii_fut_stk_cot=rng.uniform(0, 100, n))
    train, test = frame(500, 0.0), frame(500, 5.0)
    comp = Composite().fit(train)
    p_before = dict(comp.params)
    s = comp.score(test)
    assert comp.params == p_before                        # scoring never refits
    assert s["delivery_z"].mean() > 2                     # shifted test data is NOT re-centred
    assert s["delivery_z"].max() <= 3.0
