"""Divergence inputs: sign disagreement between a positioning change and the price change over the
same 20-day window, with magnitude |delta COT|. Stock-level divergence lives in positioning.py
(oi_div_mag); this module builds the market-level version against the Nifty 500."""
from __future__ import annotations

import numpy as np
import pandas as pd


def market_divergence(mkt: pd.DataFrame, nifty500_ret20: pd.Series, cfg: dict) -> pd.DataFrame:
    """mkt: output of market_positioning (date + *_cot_d20 columns). nifty500_ret20 indexed by date."""
    out = pd.DataFrame({"date": mkt["date"]})
    r = nifty500_ret20.reindex(mkt["date"]).to_numpy()
    for col in [c for c in mkt.columns if c.endswith("_cot_d20")]:
        d = mkt[col].to_numpy()
        dis = (np.sign(d) * np.sign(r)) < 0
        val = np.where(np.isfinite(d) & np.isfinite(r), dis * np.abs(d), np.nan)
        out[col.replace("_cot_d20", "_div_mag")] = val
    return out
