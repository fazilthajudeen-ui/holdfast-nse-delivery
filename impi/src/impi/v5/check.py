"""v5 Evaluation 1: the variant chosen on 2015-2022 (v5_dev.json), run ONCE on 2023-2026
(pre_registered_v5.md). The period was already used by v1-v4, so the result is POST-HOC
("rule chosen in advance, previously used period"). A lock file prevents a second run.

    python -m impi.v5.check
"""
from __future__ import annotations

import datetime as dt
import json
import sys

import numpy as np
import pandas as pd

from impi.backtest.metrics import deflated_sharpe, portfolio_metrics, spa_tests
from impi.common import METRICS, RESULTS, RESULTS_LOG, config, git_commit, log, log_result, save_json, seed
from impi.pipeline.ladder import load_closes
from impi.v5.dev import DEV_FILE, assert_preregistered_v5, calendar, nifty500_returns, run_variant
from impi.v5.strategy import BASELINES, load_events

LOCK = RESULTS / "v5_check_lock.json"
CHECK_FILE = METRICS / "v5_check.json"
SPA_MAX_P, DSR_MIN = 0.10, 0.95


def yearly(r: pd.Series) -> dict:
    return {int(y): float((1 + g).prod() - 1) for y, g in r.dropna().groupby(r.dropna().index.year)}


def main():
    reg = assert_preregistered_v5()
    dev = json.loads(DEV_FILE.read_text(encoding="utf-8"))
    if LOCK.exists():
        sys.exit(f"v5 check already run ({json.loads(LOCK.read_text())['started']}); it may only run once.")
    LOCK.write_text(json.dumps({"started": dt.datetime.now().isoformat(timespec="seconds"),
                                "git_commit": git_commit(), "chosen": dev["chosen"]}), encoding="utf-8")
    rf = float(config()["costs"]["risk_free_annual"])
    chosen = dev["chosen"]
    ev = load_events()
    ev = ev[ev["split"] == "test"]
    closes = load_closes(set(ev["symbol"]))
    cal = calendar()
    res, rets = {}, {}
    for v in [chosen] + [b for b in BASELINES if b != chosen]:
        res[v], rets[v] = run_variant(ev, v, "test", closes, cal, rf)
        log_result("V5-check", "backtest", v, "test", res[v].get("n_trades", 0), "sharpe", res[v].get("sharpe"),
                   {k: res[v].get(k) for k in ("cagr", "max_dd", "profit_factor", "event_success_rate")},
                   seed_value=seed(500))
    r = rets[chosen]
    bench = nifty500_returns().reindex(r.index)
    bm = portfolio_metrics(bench.dropna(), rf)
    spa = spa_tests(bench, pd.DataFrame({chosen: r}), int(config()["models"]["spa"]["reps"]),
                    int(config()["models"]["spa"]["block_size"]), seed(503))
    dev_val = pd.DataFrame(dev["all"])
    sr_trials = pd.to_numeric(dev_val.loc[dev_val["split"] == "validation", "sharpe_daily"], errors="coerce").dropna()
    n_trials = int(len(pd.read_csv(RESULTS_LOG)))
    m = res[chosen]
    dsr = deflated_sharpe(m.get("sharpe_daily", np.nan), m.get("days", 0), m.get("skew", 0.0), m.get("kurtosis", 3.0),
                          float(sr_trials.var(ddof=1)), n_trials)
    beats = bool(np.isfinite(m.get("sharpe", np.nan)) and m["sharpe"] > bm.get("sharpe", np.inf))
    supported = bool(beats and spa.get("spa_p_consistent", 1.0) <= SPA_MAX_P and dsr.get("dsr", 0.0) >= DSR_MIN)
    out = {"label": "POST-HOC: rule chosen on 2015-2022 before this run (pre_registered_v5.md); 2023-2026 was "
                    "already used by v1-v4, so this is not a clean test",
           "pre_registration": reg, "git_commit": git_commit(), "chosen": chosen,
           "window_events": [str(ev["date"].min().date()), str(ev["date"].max().date())],
           "results": res, "nifty500_same_days": bm, "spa_vs_nifty500": spa,
           "deflated_sharpe": {**dsr, "n_trials": n_trials, "var_trial_sr_daily": float(sr_trials.var(ddof=1))},
           "yearly_return": {"strategy": yearly(r), "nifty500": yearly(bench)},
           "pass_rule": f"Sharpe above Nifty 500 AND SPA p <= {SPA_MAX_P} AND deflated Sharpe >= {DSR_MIN}",
           "beats_nifty500_sharpe": beats, "supported": supported,
           "finished": dt.datetime.now().isoformat(timespec="seconds")}
    save_json(out, CHECK_FILE)
    for v, x in res.items():
        log(f"  {v:20s} trades {x.get('n_trades')}  CAGR {x.get('cagr', np.nan):+.3f}  Sharpe {x.get('sharpe', np.nan):+.2f}"
            f"  MaxDD {x.get('max_dd', np.nan):.3f}  PF {x.get('profit_factor', np.nan):.2f}"
            f"  success {x.get('event_success_rate', np.nan):.3f}")
    log(f"  Nifty 500 CAGR {bm.get('cagr', np.nan):+.3f} Sharpe {bm.get('sharpe', np.nan):+.2f}; SPA p "
        f"{spa.get('spa_p_consistent')}; DSR {dsr.get('dsr')}; supported {supported}")


if __name__ == "__main__":
    main()
