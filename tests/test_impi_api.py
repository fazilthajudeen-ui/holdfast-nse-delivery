"""IMPI endpoints and pages in the Delivery Lens app (skipped until `python -m impi.live` has exported data)."""
import math

import pytest
from fastapi.testclient import TestClient

from app.impi_api import APP_DIR
from app.server import app

pytestmark = pytest.mark.skipif(not (APP_DIR / "meta.json").exists(), reason="IMPI live export not built")
client = TestClient(app)


def _finite(obj):
    """No NaN/inf may leak into JSON (browsers reject them)."""
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_finite(v) for v in obj)
    return True


def test_pages_served():
    for page in ["/scanner", "/market", "/research", "/terminal"]:
        r = client.get(page)
        assert r.status_code == 200 and "<html" in r.text.lower(), page


def test_meta_and_scanner():
    m = client.get("/api/impi/meta").json()
    assert m["available"] is True and m["n_stocks"] > 100 and "label_10" in m["models"]
    s = client.get("/api/impi/scanner").json()
    assert len(s["rows"]) == m["n_stocks"] and _finite(s)
    row = s["rows"][0]
    for k in ["symbol", "p_breakout", "p_outperform", "p_vol_expansion", "xs_pct", "events"]:
        assert k in row
    assert all(0 <= r["p_breakout"] <= 1 for r in s["rows"] if r["p_breakout"] is not None)
    assert "explain" not in row


def test_stock_covered_and_not_covered():
    s = client.get("/api/impi/scanner").json()
    sym = s["rows"][0]["symbol"]
    d = client.get(f"/api/impi/stock/{sym}").json()
    assert d["covered"] and d["row"]["symbol"] == sym and len(d["history"]) > 50 and _finite(d)
    assert "label_10" in d["row"]["explain"]
    nd = client.get("/api/impi/stock/NOTAREALSYMBOL").json()
    assert nd["covered"] is False


def test_market_and_research():
    m = client.get("/api/impi/market").json()
    assert "fii_fut_idx_share" in m["series"] and len(m["series"]["fii_fut_idx_share"]) > 200 and _finite(m)
    assert "nifty500" in m["index"]
    r = client.get("/api/impi/research").json()
    assert len(r["v1_confirmatory"]) == 8 and r["models"] and _finite(r)
    assert any(f.startswith("fig1") for f in r["figures"])
    assert client.get(f"/impi-figures/{r['figures'][0]}").status_code == 200
