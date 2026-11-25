"""Benchmark ladder + backtest helpers shared by P4 (train/validation) and P6 (test, once).

Rungs (Tier 1, eligible events):  B0 base rate (all Tier-1 stock-days) -> B1 all 20-day-high breakouts
-> B2 STRICT VCP breakouts -> B3 STRICT with composite >= 1.0 -> B4 STRICT with ML probability in the
top decile (threshold = 90th percentile of validation BROAD predictions). A parallel BROAD ladder is
reported because STRICT can be small. Each rung is compared with the events of the previous rung that
it EXCLUDES (does the extra filter add value?): success-rate z-test + month-block bootstrap of the
difference in mean triple-barrier return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.backtest.executor import run_backtest
from impi.backtest.metrics import portfolio_metrics, trade_metrics
from impi.common import config, log_result, seed
from impi.research.stats import month_block_bootstrap, two_proportion_z
from impi.transforms import pit_store as pit


def rung_stats(d: pd.DataFrame) -> dict:
    d = d[d["label_10"].notna()]
    return {"n": int(len(d)), "success_rate": float(d["label_10"].mean()) if len(d) else np.nan,
            "mean_ret_tb_10": float(d["ret_tb_10"].mean()) if len(d) else np.nan,
            "mean_fwd_ex_10": float(d["fwd_ex_10"].mean()) if len(d) else np.nan}


def compare(a: pd.DataFrame, b: pd.DataFrame, rng, n_boot: int) -> dict:
    """a = rung events, b = previous-rung events excluded by the new filter."""
    a, b = a[a["label_10"].notna()], b[b["label_10"].notna()]
    z = two_proportion_z(int(a["label_10"].sum()), len(a), int(b["label_10"].sum()), len(b))
    both = pd.concat([a.assign(_g=True), b.assign(_g=False)], ignore_index=True)
    f = lambda d: float(d.loc[d["_g"], "ret_tb_10"].mean() - d.loc[~d["_g"], "ret_tb_10"].mean()) \
        if d["_g"].any() and (~d["_g"]).any() else np.nan
    bt = month_block_bootstrap(both, f, n_boot, rng) if len(a) > 5 and len(b) > 5 else {"estimate": np.nan, "p": np.nan}
    return {"success_diff": z["diff"], "success_p": z["p"], "ret_diff": bt["estimate"], "ret_p": bt.get("p"),
            "ret_ci": [bt.get("ci_low"), bt.get("ci_high")]}


def _key(d: pd.DataFrame) -> pd.Series:
    return d["symbol"].astype(str) + "|" + pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")


def ladder(rungs: list[tuple[str, pd.DataFrame]], rng, n_boot: int) -> list[dict]:
    out = []
    for i, (name, d) in enumerate(rungs):
        row = {"rung": name, **rung_stats(d)}
        if i > 0:
            prev = rungs[i - 1][1]
            excluded = prev[~_key(prev).isin(set(_key(d)))]
            row["vs_previous"] = compare(d, excluded, rng, n_boot)
            row["vs_previous"]["n_excluded"] = int(excluded["label_10"].notna().sum())
        out.append(row)
    return out


def load_closes(symbols) -> dict[str, pd.Series]:
    e = pit.read("equity_adj", columns=["date", "symbol", "close"], symbols=list(symbols))
    e["symbol"] = e["symbol"].astype(str)
    return {s: g.set_index("date")["close"] for s, g in e.groupby("symbol")}


def signals_from(ev: pd.DataFrame, priority: pd.Series | None = None) -> pd.DataFrame:
    s = ev[["date", "symbol", "family", "entry_date", "exit_date_10", "entry_px", "exit_px_10", "liq_bucket"]].copy()
    s = s.rename(columns={"exit_date_10": "exit_date", "exit_px_10": "exit_px"})
    s["priority"] = priority.loc[ev.index].to_numpy() if priority is not None else 0.0
    return s


def backtest_variant(name: str, ev: pd.DataFrame, closes, calendar, split: str, phase: str,
                     priority: pd.Series | None = None) -> tuple[dict, pd.Series]:
    rf = float(config()["costs"]["risk_free_annual"])
    ret, tr = run_backtest(signals_from(ev, priority), closes, calendar, seed=seed(400))
    pm = portfolio_metrics(ret, rf) if len(ret) else {}
    tm = trade_metrics(tr)
    if len(tr):
        tm["by_bucket"] = tr.groupby(tr["bucket"].fillna("small"))["net_ret"].agg(["count", "mean"]).round(5).to_dict("index")
    pm["turnover_per_year"] = ret.attrs.get("turnover_per_year") if len(ret) else None
    res = {"variant": name, "split": split, **pm, **{k: v for k, v in tm.items()},
           "label": "ok" if tm.get("n_trades", 0) >= 30 else "insufficient trades (n<30)"}
    log_result(phase, "backtest", name, split, tm.get("n_trades", 0), "sharpe", pm.get("sharpe"),
               {k: res.get(k) for k in ("cagr", "max_dd", "sortino", "calmar", "hit_rate", "profit_factor")},
               seed_value=seed(400))
    return res, ret
