"""Algorithmic VCP (volatility contraction pattern) detector (spec section 5, config features.yaml: vcp).

Definitions (D-014):
  * Fractal pivots with k bars each side: pivot high at i if high[i] > max(high[i-k:i]) and
    high[i] >= max(high[i+1:i+k+1]) (mirror for lows). A pivot at i is CONFIRMED only at bar i+k,
    so a pattern evaluated for day t uses pivots with i + k <= t - 1 (G1).
  * Base: within the last base_max_days bars before t, the highest pivot high H1 starts the base
    (t - idx(H1) >= base_min_days). After H1, consecutive same-type pivots are merged (highest high /
    lowest low) and the sequence must alternate H1, L1, H2, L2, ..., Hn, Ln (it must END on a low).
  * Valid when 2 <= n <= 4, contraction depths d_i = (H_i - L_i) / H_i are strictly declining and each
    < max_depth (50%), and the average volume of each contraction (bars from H_i to L_i) strictly
    declines. Pivot = H_n. The pattern is void if any close after H_n (up to t-1) is already above
    pivot * pivot_buffer.
  * STRICT trigger on day t: close_t > pivot * 1.001 and volume_t >= 1.5 x mean(volume[t-20:t]).
"""
from __future__ import annotations

import numpy as np

EMPTY = {"vcp_valid": 0.0, "vcp_n": 0.0, "vcp_depth_first": np.nan, "vcp_depth_last": np.nan,
         "vcp_base_len": np.nan, "vcp_vol_trend": np.nan, "vcp_pivot": np.nan}


def fractal_pivots(high: np.ndarray, low: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays (is_pivot_high, is_pivot_low) at the pivot bar itself (not the confirm bar)."""
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    n = len(high)

    def shifted(a, s):
        out = np.full(n, np.nan)
        if s > 0:
            out[s:] = a[:-s]
        else:
            out[:s] = a[-s:]
        return out

    lmax = np.fmax.reduce([shifted(high, s) for s in range(1, k + 1)])
    rmax = np.fmax.reduce([shifted(high, -s) for s in range(1, k + 1)])
    lmin = np.fmin.reduce([shifted(low, s) for s in range(1, k + 1)])
    rmin = np.fmin.reduce([shifted(low, -s) for s in range(1, k + 1)])
    with np.errstate(invalid="ignore"):
        ph = np.isfinite(high) & (high > lmax) & (high >= rmax)
        pl = np.isfinite(low) & (low < lmin) & (low <= rmin)
    ph[:k] = ph[n - k:] = False
    pl[:k] = pl[n - k:] = False
    return ph, pl


class VCPDetector:
    def __init__(self, high, low, close, volume, cfg: dict):
        self.h = np.asarray(high, float)
        self.l = np.asarray(low, float)
        self.c = np.asarray(close, float)
        self.v = np.asarray(volume, float)
        self.cfg = cfg
        self.k = int(cfg["fractal_k"])
        ph, pl = fractal_pivots(self.h, self.l, self.k)
        # pivots as (index, type, price), type +1 high / -1 low; sorted by index
        idx = np.flatnonzero(ph | pl)
        piv = []
        for i in idx:
            if ph[i]:
                piv.append((i, 1, self.h[i]))
            if pl[i]:
                piv.append((i, -1, self.l[i]))
        self.piv_idx = np.array([p[0] for p in piv], int)
        self.piv = piv
        v = np.where(np.isfinite(self.v), self.v, 0.0)
        self.vcum = np.concatenate([[0.0], np.cumsum(v)])
        self.vcnt = np.concatenate([[0.0], np.cumsum(np.isfinite(self.v))])

    def _avg_vol(self, a: int, b: int) -> float:
        n = self.vcnt[b + 1] - self.vcnt[a]
        return (self.vcum[b + 1] - self.vcum[a]) / n if n > 0 else np.nan

    def pattern_at(self, t: int) -> dict:
        """Pattern formed BEFORE day t (uses pivots confirmed by t-1)."""
        cfg, k = self.cfg, self.k
        lo_bar = t - int(cfg["base_max_days"])
        hi_bar = t - 1 - k                                  # last pivot index confirmed by t-1
        if hi_bar <= lo_bar or lo_bar < 0:
            return dict(EMPTY)
        a = np.searchsorted(self.piv_idx, lo_bar, side="left")
        b = np.searchsorted(self.piv_idx, hi_bar, side="right")
        seq = self.piv[a:b]
        highs = [p for p in seq if p[1] == 1]
        if not highs:
            return dict(EMPTY)
        h1 = max(highs, key=lambda p: (p[2], -p[0]))       # highest high, earliest if tied
        if t - h1[0] < int(cfg["base_min_days"]):
            return dict(EMPTY)
        rest = [p for p in seq if p[0] >= h1[0] and not (p[0] == h1[0] and p[1] == -1)]
        merged = []
        for p in rest:
            if merged and merged[-1][1] == p[1]:
                q = merged[-1]
                if (p[1] == 1 and p[2] > q[2]) or (p[1] == -1 and p[2] < q[2]):
                    merged[-1] = p
            else:
                merged.append(p)
        if not merged or merged[0][1] != 1 or merged[-1][1] != -1:
            return dict(EMPTY)
        pairs = [(merged[i], merged[i + 1]) for i in range(0, len(merged) - 1, 2)]
        n = len(pairs)
        if n < int(cfg["min_contractions"]) or n > int(cfg["max_contractions"]):
            return dict(EMPTY)
        depths = [(H[2] - L[2]) / H[2] for H, L in pairs]
        vols = [self._avg_vol(H[0], L[0]) for H, L in pairs]
        if any(d >= cfg["max_depth"] or d <= 0 for d in depths):
            return dict(EMPTY)
        if any(depths[i] <= depths[i + 1] for i in range(n - 1)):
            return dict(EMPTY)
        if any(not (vols[i] > vols[i + 1]) for i in range(n - 1)):
            return dict(EMPTY)
        pivot = pairs[-1][0][2]
        after = self.c[pairs[-1][0][0] + 1:t]
        if after.size and np.nanmax(np.where(np.isfinite(after), after, -np.inf)) > pivot * cfg["pivot_buffer"]:
            return dict(EMPTY)
        return {"vcp_valid": 1.0, "vcp_n": float(n), "vcp_depth_first": depths[0],
                "vcp_depth_last": depths[-1], "vcp_base_len": float(t - h1[0]),
                "vcp_vol_trend": vols[-1] / vols[0] if vols[0] > 0 else np.nan, "vcp_pivot": pivot}

    def triggers(self, vol_ratio20: np.ndarray) -> list[tuple[int, dict]]:
        """All STRICT trigger days t with their pattern."""
        out = []
        cand = np.flatnonzero(np.asarray(vol_ratio20, float) >= self.cfg["volume_mult"])
        for t in cand:
            if t < 1 or not np.isfinite(self.c[t]):
                continue
            p = self.pattern_at(int(t))
            if p["vcp_valid"] == 1.0 and self.c[t] > p["vcp_pivot"] * self.cfg["pivot_buffer"]:
                out.append((int(t), p))
        return out
