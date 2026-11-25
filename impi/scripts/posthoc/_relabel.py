"""Re-labelling helper for the POST-HOC spec curve (C) and tie analysis (G).

Uses the FROZEN functions unchanged: compression.wilder_atr (ATR14 on the split-adjusted series) and
labels.triple_barrier (barrier logic, same-day tie -> loss, gap fills, D-017 stop-trading rule). Only the
barrier multiples / horizon passed in differ. Validation: with (2.0, 1.0, 10) the re-computed labels are
compared with the frozen label_10 and the agreement is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.features.compression import wilder_atr
from impi.labels.triple_barrier import triple_barrier
from impi.transforms import pit_store as pit


class Prices:
    def __init__(self, symbols):
        e = pit.read("equity_adj", columns=["date", "symbol", "series", "open", "high", "low", "close"],
                     symbols=sorted(set(map(str, symbols))))
        e["symbol"] = e["symbol"].astype(str)
        e = e.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.data_end = e["date"].max()
        self.s = {}
        for s, g in e.groupby("symbol", sort=False):
            g = g.reset_index(drop=True)
            self.s[s] = {"date": g["date"].to_numpy(), "o": g["open"].to_numpy(float), "h": g["high"].to_numpy(float),
                         "l": g["low"].to_numpy(float), "c": g["close"].to_numpy(float),
                         "atr": wilder_atr(g, 14).to_numpy(float),
                         "stops": bool(g["date"].iloc[-1] < self.data_end)}

    def label(self, ev: pd.DataFrame, up: float, lo: float, horizon: int) -> pd.DataFrame:
        """Returns label, ret, exit_k, label_end, tie (both barriers on the resolving bar), gap and ATR measures."""
        parts = []
        for s, g in ev.groupby(ev["symbol"].astype(str), sort=False):
            p = self.s.get(s)
            if p is None:
                continue
            t = np.searchsorted(p["date"], g["date"].to_numpy(dtype="datetime64[ns]"))
            ok = (t < len(p["date"])) & (p["date"][np.minimum(t, len(p["date"]) - 1)] == g["date"].to_numpy(dtype="datetime64[ns]"))
            t = np.where(ok, t, 0)
            r = triple_barrier(p["o"], p["h"], p["l"], p["c"], p["atr"], t, horizon, up, lo, p["stops"])
            n = len(p["c"])
            ek = r["exit_k"]
            kk = np.where(np.isfinite(ek), t + np.nan_to_num(ek, nan=0).astype(int), t + horizon)
            label_end = np.where(kk < n, p["date"][np.minimum(kk, n - 1)], np.datetime64("NaT"))
            # tie: on the resolving bar both barriers were touched (frozen rule counts it as a loss)
            j = np.minimum(t + np.nan_to_num(ek, nan=1).astype(int), n - 1)
            res_bar = np.isfinite(ek) & (ek >= 1)
            tie = res_bar & (p["l"][j] <= r["lower"]) & (p["h"][j] >= r["upper"]) & (r["label"] == 0)
            t1 = np.minimum(t + 1, n - 1)
            gap = p["o"][t1] / p["c"][t] - 1
            parts.append(pd.DataFrame({"_idx": g.index[ok], "label": r["label"][ok], "ret": r["ret"][ok], "exit_k": ek[ok],
                                       "new_label_end": label_end[ok], "tie": tie[ok],
                                       "gap_pct": 100 * gap[ok], "gap_atr": ((p["o"][t1] - p["c"][t]) / p["atr"][t])[ok],
                                       "atr_pct": (100 * p["atr"][t] / p["c"][t])[ok],
                                       "day1_range_atr": ((p["h"][t1] - p["l"][t1]) / p["atr"][t])[ok]}))
        return pd.concat(parts).set_index("_idx").reindex(ev.index)
