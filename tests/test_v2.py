"""IMPI v2: extra features have no look-ahead; scoring is row-wise (batch-independent); live config guard."""
import numpy as np
import pandas as pd

from impi.v2.features_v2 import extra_features
from tests.test_features_labels import synthetic_stock


def test_extra_features_no_look_ahead():
    df = synthetic_stock(n=500, seed=4)
    full = extra_features(df)
    for cut in [260, 377, 499]:
        part = extra_features(df.iloc[:cut].copy())
        pd.testing.assert_frame_equal(part, full.iloc[:cut], check_exact=False, rtol=1e-9, atol=1e-12)


def test_extra_feature_ranges():
    f = extra_features(synthetic_stock(n=500, seed=5))
    assert (f["dist_52w_high"].dropna() <= 1e-12).all() and (f["dist_52w_low"].dropna() >= -1e-12).all()
    assert f["clv5"].dropna().between(0, 1).all()
    assert f["rv20_pct252"].dropna().between(0, 100).all()


def test_scoring_is_row_wise():
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from impi.v2.train import score
    r = np.random.default_rng(0)
    X = pd.DataFrame(r.normal(size=(400, 3)), columns=["a", "b", "c"])
    y = (X["a"] + r.normal(size=400) > 0).astype(int)
    model = {"cols": ["a", "b", "c"], "parts": {"xgb": XGBClassifier(n_estimators=20, max_depth=2).fit(X, y),
                                                "logit": LogisticRegression().fit(X, y)}}
    batch = score(model, X)
    single = np.array([score(model, X.iloc[[i]])[0] for i in range(25)])
    np.testing.assert_allclose(batch[:25], single, rtol=1e-6)


def test_live_end_override_only_when_outputs_redirected(monkeypatch):
    import impi.common as c
    monkeypatch.setenv("IMPI_DATA_END", "2030-01-01")
    c.config.cache_clear()
    try:
        assert str(c.config()["data"]["end"]) == "2026-10-09"      # research folders -> frozen end date kept
    finally:
        c.config.cache_clear()
