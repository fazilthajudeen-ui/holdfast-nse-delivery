"""Rule-based composite score (spec section 8). Weights fixed a priori in config/models.yaml.

  CompressionQuality    = z( mean( -z(bbw_pct252), -z(atr_ratio), z(nr7_count10), -z(range_contraction) ) )
  DeliveryZ             = z( deliv_z20 )
  PositioningAlignment  = z( 0.5 * z(price_oi_align) + 0.5 * (fii_fut_stk_cot - 50) / 28.87 )
                          (stock OI confirming the price move + FII stock-futures positioning high in
                          its 52-week range = supportive accumulation scores high; D-019)
  VolumeAnomaly         = z( vol_z )
Every z uses the mean/std of the TRAIN-split reference events only, then clips to [-3, 3].
Missing components count as 0 (neutral). Score = 0.30 CQ + 0.25 DZ + 0.25 PA + 0.20 VA.
Tier 2 (no positioning) uses tier2_weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.common import config

UNIFORM_SD = 100 / np.sqrt(12)        # sd of a uniform 0-100 variable, scales a COT index to ~unit variance
SUB = {"cq": [("bbw_pct252", -1), ("atr_ratio", -1), ("nr7_count10", 1), ("range_contraction", -1)]}


class Composite:
    def __init__(self):
        self.params: dict[str, tuple[float, float]] = {}
        c = config()["models"]["composite"]
        self.w = c["weights"]
        self.w2 = c["tier2_weights"]
        self.clip = float(c["clip"])
        self.threshold = float(c["threshold"])

    def _z(self, name: str, x: pd.Series, fit: bool) -> pd.Series:
        x = x.astype(float)
        if fit:
            mu, sd = float(x.mean()), float(x.std(ddof=1))
            self.params[name] = (mu, sd if sd > 0 else 1.0)
        mu, sd = self.params[name]
        return ((x - mu) / sd).clip(-self.clip, self.clip)

    def _components(self, ev: pd.DataFrame, fit: bool) -> pd.DataFrame:
        out = pd.DataFrame(index=ev.index)
        cq_parts = [sign * self._z(f"cq_{col}", ev[col], fit) for col, sign in SUB["cq"]]
        cq_raw = pd.concat(cq_parts, axis=1).mean(axis=1, skipna=True)
        out["compression_quality"] = self._z("cq", cq_raw, fit)
        out["delivery_z"] = self._z("dz", ev["deliv_z20"], fit)
        pa_stock = self._z("pa_stock", ev["price_oi_align"], fit)
        pa_mkt = (ev["fii_fut_stk_cot"].astype(float) - 50) / UNIFORM_SD
        pa_raw = pd.concat([pa_stock, pa_mkt], axis=1).mean(axis=1, skipna=True)
        out["positioning_alignment"] = self._z("pa", pa_raw, fit)
        out["volume_anomaly"] = self._z("va", ev["vol_z"], fit)
        return out.fillna(0.0)

    def fit(self, train_events: pd.DataFrame) -> "Composite":
        self._components(train_events, fit=True)
        return self

    def score(self, ev: pd.DataFrame, tier2: bool = False) -> pd.DataFrame:
        comp = self._components(ev, fit=False)
        w = self.w2 if tier2 else self.w
        comp["score"] = sum(comp[k] * v for k, v in w.items())
        comp["tradeable"] = comp["score"] >= self.threshold
        return comp
