"""IMPI v3 option-implied features: per-day, per-stock implied volatility from the F&O bhavcopy.

    python -m impi.v3.options            # build/refresh the per-day cache and the PIT table

For every trading day and every stock with traded options:
  * forward F for an expiry = that expiry's stock-future close (same-day, same expiry, raw prices, so no
    corporate-action adjustment is needed: IV is scale-free);
  * only options that actually TRADED that day are used (untraded contracts carry stale closes);
  * implied volatility by Black-76 (r = 6.5%, act/365 calendar time), solved by vectorised bisection;
  * OTM convention: calls for K >= F, puts for K < F;
  * ATM IV per expiry = linear interpolation of OTM IVs in log-moneyness at k = 0;
  * iv30 = constant-maturity 30-calendar-day ATM IV (total-variance interpolation between the near and
    next expiries; near is skipped within 3 days of expiry);
  * wings on the main expiry (near if >= 7 days left, else next): put wing = mean IV, k in [-0.15, -0.05];
    call wing = mean IV, k in [0.05, 0.15];
  * traded contracts: calls, puts, futures (lot-size free ratios are formed downstream).

The F&O bhavcopy is released T 19:30 (data/release_lag.yaml), before the T 23:59 decision point.
"""
from __future__ import annotations

import datetime as dt
import io
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import ndtr

from impi.common import DATA, config, log
from impi.ingest import corp_actions as CA
from impi.ingest.download import raw_path
from impi.ingest.nse_bhav import unzip
from impi.transforms import pit_store as pit

R = 0.065
CACHE = DATA / "processed" / "cache" / "opt_iv"          # shared by research and live (pure function of one raw file)
VERSION = 1
OUT_COLS = ["date", "symbol", "iv_near", "iv_next", "iv30", "dte_near", "dte_next", "iv_put_wing", "iv_call_wing",
            "n_iv_main", "call_contracts", "put_contracts", "fut_contracts"]


# ------------------------------------------------------------------ parsing (only what IV needs)
def _legacy(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(unzip(content)), usecols=["INSTRUMENT", "SYMBOL", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP",
                                                         "CLOSE", "CONTRACTS", "TIMESTAMP"], dtype=str)
    df = df.apply(lambda s: s.str.strip())
    df = df[df["INSTRUMENT"].isin(["FUTSTK", "OPTSTK"])]
    return pd.DataFrame({
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y"), "kind": df["INSTRUMENT"].str[:3],
        "symbol": df["SYMBOL"], "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y"),
        "strike": pd.to_numeric(df["STRIKE_PR"], errors="coerce"), "opt": df["OPTION_TYP"],
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"), "contracts": pd.to_numeric(df["CONTRACTS"], errors="coerce")})


def _udiff(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(unzip(content)), usecols=["TradDt", "FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric",
                                                         "OptnTp", "ClsPric", "TtlTradgVol"], dtype=str)
    df = df.apply(lambda s: s.str.strip())
    df = df[df["FinInstrmTp"].isin(["STF", "STO"])]
    return pd.DataFrame({
        "date": pd.to_datetime(df["TradDt"], format="%Y-%m-%d"),
        "kind": df["FinInstrmTp"].map({"STF": "FUT", "STO": "OPT"}), "symbol": df["TckrSymb"],
        "expiry": pd.to_datetime(df["XpryDt"], format="%Y-%m-%d"),
        "strike": pd.to_numeric(df["StrkPric"], errors="coerce"), "opt": df["OptnTp"],
        "close": pd.to_numeric(df["ClsPric"], errors="coerce"), "contracts": pd.to_numeric(df["TtlTradgVol"], errors="coerce")})


# ------------------------------------------------------------------ Black-76
def black76(F, K, T, sig, call):
    sq = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sig * sig * T) / sq
    d2 = d1 - sq
    df = np.exp(-R * T)
    c = df * (F * ndtr(d1) - K * ndtr(d2))
    p = df * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(call, c, p)


def implied_vol(price, F, K, T, call, lo=0.01, hi=4.0, iters=60):
    """Vectorised bisection; NaN where the price is outside the no-arbitrage band of [lo, hi] vol."""
    price, F, K, T, call = map(np.asarray, (price, F, K, T, call))
    plo, phi = black76(F, K, T, lo, call), black76(F, K, T, hi, call)
    ok = (price > plo) & (price < phi) & (T > 0) & (F > 0) & (K > 0)
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = black76(F, K, T, m, call) > price
        b = np.where(up, m, b)
        a = np.where(up, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


# ------------------------------------------------------------------ per-day features
def _atm(k: np.ndarray, iv: np.ndarray) -> float:
    """IV at k = 0 by linear interpolation between the nearest OTM strikes on each side."""
    lo, hi = k < 0, k >= 0
    if lo.any() and hi.any():
        i, j = np.flatnonzero(lo)[np.argmax(k[lo])], np.flatnonzero(hi)[np.argmin(k[hi])]
        if k[j] - k[i] <= 0.15:
            return float(iv[i] + (iv[j] - iv[i]) * (0 - k[i]) / (k[j] - k[i]))
    near = np.abs(k) <= 0.03
    return float(iv[near][np.argmin(np.abs(k[near]))]) if near.any() else np.nan


def day_features(c: pd.DataFrame) -> pd.DataFrame:
    if c.empty:
        return pd.DataFrame(columns=OUT_COLS)
    d = c["date"].iloc[0]
    fut = c[(c["kind"] == "FUT")]
    vol = c[c["contracts"] > 0].groupby(["symbol", "kind", "opt"], dropna=False)["contracts"].sum()
    fwd = fut[fut["close"] > 0].set_index(["symbol", "expiry"])["close"]
    o = c[(c["kind"] == "OPT") & (c["contracts"] > 0) & (c["close"] > 0) & c["opt"].isin(["CE", "PE"])].copy()
    o = o.join(fwd.rename("F"), on=["symbol", "expiry"]).dropna(subset=["F", "strike"])
    o["k"] = np.log(o["strike"] / o["F"])
    o = o[((o["opt"] == "CE") & (o["k"] >= 0)) | ((o["opt"] == "PE") & (o["k"] < 0))]     # OTM side only
    o = o[o["k"].abs() <= 0.3]
    o["dte"] = (o["expiry"] - d).dt.days
    o["T"] = np.maximum(o["dte"], 0.5) / 365.0
    o["iv"] = implied_vol(o["close"].to_numpy(), o["F"].to_numpy(), o["strike"].to_numpy(), o["T"].to_numpy(),
                          (o["opt"] == "CE").to_numpy())
    o = o.dropna(subset=["iv"]).sort_values(["symbol", "expiry"], kind="stable")
    syms_o = o["symbol"].to_numpy()
    exp_o, dte_o = o["expiry"].to_numpy(), o["dte"].to_numpy()
    k_o, iv_o = o["k"].to_numpy(), o["iv"].to_numpy()
    starts = np.flatnonzero(np.r_[True, syms_o[1:] != syms_o[:-1]]) if len(o) else np.array([], int)
    ends = np.r_[starts[1:], len(o)]
    rows = []
    for a, b in zip(starts, ends):
        sym = syms_o[a]
        ex = exp_o[a:b]
        e0 = np.flatnonzero(np.r_[True, ex[1:] != ex[:-1]])
        e1 = np.r_[e0[1:], b - a]
        per = [(ex[c0], int(dte_o[a + c0]), _atm(k_o[a + c0:a + c1], iv_o[a + c0:a + c1]))
               for c0, c1 in list(zip(e0, e1))[:3]]
        usable = [(e, dte, iv) for e, dte, iv in per if dte >= 3 and np.isfinite(iv)]
        near = usable[0] if usable else (None, np.nan, np.nan)
        nxt = usable[1] if len(usable) > 1 else (None, np.nan, np.nan)
        iv30 = np.nan
        if np.isfinite(near[2]) and np.isfinite(nxt[2]):
            t1, t2 = near[1] / 365, nxt[1] / 365
            w1, w2 = near[2] ** 2 * t1, nxt[2] ** 2 * t2
            t = 30 / 365
            w = w1 + (w2 - w1) * (t - t1) / (t2 - t1)
            if w > 0:
                iv30 = float(np.sqrt(w / t))
        elif np.isfinite(near[2]):
            iv30 = near[2]
        main = near[0] if (near[0] is not None and near[1] >= 7) else (nxt[0] if nxt[0] is not None else near[0])
        sel = (ex == main) if main is not None else np.zeros(b - a, bool)
        km, ivm = k_o[a:b][sel], iv_o[a:b][sel]
        pw = ivm[(km >= -0.15) & (km <= -0.05)]
        cw = ivm[(km >= 0.05) & (km <= 0.15)]
        rows.append({"symbol": sym, "iv_near": near[2], "iv_next": nxt[2], "iv30": iv30, "dte_near": near[1],
                     "dte_next": nxt[1], "iv_put_wing": pw.mean() if len(pw) else np.nan,
                     "iv_call_wing": cw.mean() if len(cw) else np.nan, "n_iv_main": int(sel.sum())})
    out = pd.DataFrame(rows, columns=[c_ for c_ in OUT_COLS if c_ not in ("date", "call_contracts", "put_contracts", "fut_contracts")])
    syms = pd.Index(sorted(set(fut["symbol"]) | set(out["symbol"])), name="symbol")
    out = out.set_index("symbol").reindex(syms)

    def vsum(kind, opt):
        try:
            s = vol.xs((kind, opt), level=("kind", "opt"))
        except KeyError:
            return 0.0
        return s.reindex(syms).fillna(0.0)
    out["call_contracts"] = vsum("OPT", "CE")
    out["put_contracts"] = vsum("OPT", "PE")
    fv = vol[vol.index.get_level_values("kind") == "FUT"].groupby(level="symbol").sum()
    out["fut_contracts"] = fv.reindex(syms).fillna(0.0)
    out = out.reset_index()
    out.insert(0, "date", d)
    return out[OUT_COLS]


def for_date(d: dt.date) -> pd.DataFrame | None:
    last_legacy = config()["data"]["sources"]["fo_legacy"]["last_date"]
    order = [("fo_legacy", _legacy), ("fo_udiff", _udiff)]
    if d > last_legacy:
        order.reverse()
    for name, parser in order:
        p = raw_path(name, d)
        if p.exists():
            cache = CACHE / f"v{VERSION}_{name}_{d.isoformat()}.parquet"
            if cache.exists() and cache.stat().st_mtime >= p.stat().st_mtime:
                return pd.read_parquet(cache)
            c = parser(p.read_bytes())
            c = c[c["date"] == pd.Timestamp(d)]                    # a stale file would carry another date
            out = day_features(c)
            out["source"] = name
            cache.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(cache, index=False)
            return out
    return None


def build(workers: int = 3) -> dict:
    dates = pit.trading_dates()
    frames, missing = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (d, df) in enumerate(zip(dates, ex.map(for_date, dates, chunksize=8)), 1):
            if df is None or df.empty:
                missing.append(d.isoformat())
            else:
                frames.append(df)
            if i % 250 == 0:
                log(f"  options: {i}/{len(dates)} days")
    iv = pd.concat(frames, ignore_index=True)
    changes = pd.read_parquet(pit.REFERENCE / "symbol_changes.parquet")
    iv, _ = CA.apply_symbol_changes(iv, changes)
    iv = iv.drop_duplicates(["symbol", "date"])
    iv["available_at"] = iv["date"] + pit.release_offset("fo_bhav")
    iv.to_parquet(pit.PIT / "options_iv.parquet", index=False)
    qc = {"rows": int(len(iv)), "days": int(iv["date"].nunique()), "missing_days": missing,
          "iv30_coverage": float(iv["iv30"].notna().mean()), "iv30_median": float(iv["iv30"].median()),
          "iv30_p01_p99": [float(iv["iv30"].quantile(0.01)), float(iv["iv30"].quantile(0.99))]}
    log(f"options_iv: {qc}")
    return qc


def main():
    from impi.common import METRICS, save_json
    qc = build()
    save_json(qc, METRICS / "options_iv_qc.json")


if __name__ == "__main__":
    main()
