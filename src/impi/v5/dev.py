"""v5 development run: all 120 strategy variants on TRAIN (2015-06..2020) and VALIDATION (2021-22) only,
then the pre-registered selection rule picks ONE variant (pre_registered_v5.md).

    python -m impi.v5.dev

Selection: among variants with >= 30 trades in both periods, the highest min(train Sharpe, validation
Sharpe) after costs. The test period (2023+) is not read here.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import time

import numpy as np
import pandas as pd

from impi.backtest.executor import run_backtest
from impi.backtest.metrics import portfolio_metrics, trade_metrics
from impi.common import METRICS, ROOT, SANDBOX, config, git_commit, log, log_result, save_json, seed
from impi.features.build import OUT
from impi.pipeline.ladder import load_closes
from impi.transforms import pit_store as pit
from impi.v5.strategy import BASELINES, load_events, select, signals, success_rate, variants

PREREG = "pre_registered_v5.md"
DEV_FILE = METRICS / "v5_dev.json"
MIN_TRADES = 30
SPLITS = ["train", "validation"]


def assert_preregistered_v5() -> str:
    if SANDBOX:
        return "SANDBOX"
    if not (ROOT / PREREG).exists():
        raise SystemExit(f"{PREREG} missing - write and commit it first.")
    first = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %cI", "--", PREREG], cwd=ROOT,
                           capture_output=True, text=True).stdout.split()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG, "src/impi/v5"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if not first or dirty:
        raise SystemExit(f"{PREREG} and src/impi/v5 must be committed and unchanged before running.")
    return " ".join(first[:2])


def calendar() -> np.ndarray:
    return np.array(sorted(pd.read_parquet(OUT / "base.parquet", columns=["date"])["date"].unique()),
                    dtype="datetime64[ns]")


def nifty500_returns() -> pd.Series:
    idx = pit.read("index")
    return idx[idx["index"] == "nifty500"].set_index("date")["close"].sort_index().pct_change()


def run_variant(ev: pd.DataFrame, v: str, split: str, closes, cal, rf: float) -> tuple[dict, pd.Series]:
    d = select(ev, v)
    d = d[d["split"] == split]
    ret, tr = run_backtest(signals(d, v, closes), closes, cal, seed=seed(500))
    pm = portfolio_metrics(ret, rf) if len(ret) else {}
    tm = trade_metrics(tr)
    return {"variant": v, "split": split, "n_events": int(len(d)), "event_success_rate": success_rate(d, v),
            **pm, **tm}, ret


def main():
    reg = assert_preregistered_v5()
    t0 = time.time()
    rf = float(config()["costs"]["risk_free_annual"])
    ev = load_events()
    ev = ev[ev["split"].isin(SPLITS)]                      # the test period is never loaded into this run
    closes = load_closes(set(ev["symbol"]))
    cal = calendar()
    bench = nifty500_returns()
    rows = []
    for i, v in enumerate(variants()):
        per = {}
        for split in SPLITS:
            m, _ = run_variant(ev, v, split, closes, cal, rf)
            per[split] = m
            rows.append(m)
        score = min(per["train"].get("sharpe", np.nan), per["validation"].get("sharpe", np.nan))
        ok = all(per[s].get("n_trades", 0) >= MIN_TRADES for s in SPLITS)
        log_result("V5-dev", "backtest", v, "train+validation", per["train"].get("n_trades", 0)
                   + per["validation"].get("n_trades", 0), "sharpe_min_train_val", score,
                   {s: {k: per[s].get(k) for k in ("sharpe", "cagr", "max_dd", "profit_factor", "n_trades",
                                                   "event_success_rate")} for s in SPLITS}, seed_value=seed(500))
        if (i + 1) % 10 == 0:
            log(f"  v5 dev: {i + 1}/120 variants ({time.time() - t0:.0f}s)")
    tab = pd.DataFrame(rows)
    wide = tab.pivot(index="variant", columns="split", values=["sharpe", "n_trades"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide["eligible"] = (wide["n_trades_train"] >= MIN_TRADES) & (wide["n_trades_validation"] >= MIN_TRADES)
    wide["score"] = wide[["sharpe_train", "sharpe_validation"]].min(axis=1)
    ranked = wide[wide["eligible"]].sort_values("score", ascending=False, kind="stable")
    chosen = str(ranked.index[0])
    bm = {}
    for split in SPLITS:
        a, b = config()["splits"][split]
        r = bench[(bench.index >= pd.Timestamp(str(a))) & (bench.index <= pd.Timestamp(str(b)))].dropna()
        bm[split] = portfolio_metrics(r, rf)
    out = {"label": "development (train + validation only); selection rule fixed in pre_registered_v5.md",
           "pre_registration": reg, "git_commit": git_commit(),
           "finished": dt.datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0),
           "selection_rule": "max over variants with >= 30 trades in train and validation of "
                             "min(train Sharpe, validation Sharpe), after costs",
           "chosen": chosen, "chosen_sha256": hashlib.sha256(chosen.encode()).hexdigest()[:16],
           "chosen_metrics": {r_["split"]: r_ for r_ in rows if r_["variant"] == chosen},
           "baselines": {b: {r_["split"]: r_ for r_ in rows if r_["variant"] == b} for b in BASELINES},
           "nifty500": bm, "top10": ranked.head(10).reset_index().to_dict("records"),
           "n_variants": int(len(wide)), "n_eligible": int(wide["eligible"].sum()),
           "n_positive_both": int(((wide["sharpe_train"] > 0) & (wide["sharpe_validation"] > 0)).sum()),
           "all": rows}
    save_json(out, DEV_FILE)
    log(f"v5 dev: chosen {chosen}  score {wide.loc[chosen, 'score']:.3f}  ({out['n_positive_both']} of 120 "
        f"variants positive in both periods)")


if __name__ == "__main__":
    main()
