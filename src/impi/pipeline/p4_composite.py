"""P4 - rule-based composite baseline, benchmark ladder B0-B3 and costed backtests on TRAIN and
VALIDATION (the test set is not touched).

    python -m impi.pipeline.p4_composite
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from impi.common import METRICS, PROCESSED, config, log, rng, save_json
from impi.features.build import OUT
from impi.models.composite import Composite
from impi.pipeline.common_phase import assert_preregistered, tier1
from impi.pipeline.ladder import backtest_variant, ladder, load_closes
from impi.pipeline.p2_events import assert_frozen, load_events
from impi.validation.splits import assign_split

MODELS = PROCESSED / "models"


def main():
    assert_preregistered()
    assert_frozen()
    ev = load_events()
    comp = Composite().fit(tier1(ev, "broad", ["train"]))
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / "composite.pkl", "wb") as f:
        pickle.dump(comp, f)
    sc = comp.score(ev)
    ev = pd.concat([ev, sc.add_prefix("comp_")], axis=1)
    base = pd.read_parquet(OUT / "base.parquet")
    base["split"] = assign_split(base["date"], base["label_end"])
    cal = np.array(sorted(base["date"].unique()), dtype="datetime64[ns]")
    n_boot = int(config()["models"]["bootstrap"]["n"]) // 2
    out = {"composite_params": {k: list(v) for k, v in comp.params.items()}, "splits": {}}
    symbols = set(ev.loc[ev["tier1"], "symbol"].astype(str))
    closes = load_closes(symbols)
    for split in ["train", "validation"]:
        b0 = base[base["eligible"] & (base["split"] == split) & base["label_10"].notna()]
        b1 = tier1(ev, "all", [split])
        b2 = tier1(ev, "strict", [split])
        b3 = b2[b2["comp_tradeable"]]
        br = tier1(ev, "broad", [split])
        br3 = br[br["comp_tradeable"]]
        r = rng(400 + len(split))
        res = {"ladder_strict": ladder([("B0_base", b0), ("B1_all_breakouts", b1), ("B2_strict", b2),
                                        ("B3_strict_composite", b3)], r, n_boot),
               "ladder_broad": ladder([("B0_base", b0), ("B1_all_breakouts", b1), ("B2b_broad", br),
                                       ("B3b_broad_composite", br3)], r, n_boot),
               "composite_auc": {}}
        from sklearn.metrics import roc_auc_score
        for fam, d in [("strict", b2), ("broad", br)]:
            if d["label_10"].nunique() == 2:
                res["composite_auc"][fam] = float(roc_auc_score(d["label_10"], d["comp_score"]))
        bts, rets = [], {}
        for name, d in [("B1_all_breakouts", b1), ("B2_strict", b2), ("B3_strict_composite", b3),
                        ("B2b_broad", br), ("B3b_broad_composite", br3)]:
            if len(d) == 0:
                continue
            m, rr = backtest_variant(name, d, closes, cal, split, "P4", priority=ev["comp_score"])
            bts.append(m)
            rets[name] = rr
        res["backtests"] = bts
        pd.DataFrame(rets).to_parquet(METRICS / f"variant_returns_{split}.parquet")
        out["splits"][split] = res
        log(f"P4 {split}: ladder strict {[(x['rung'], x['n'], round(x['success_rate'], 3)) for x in res['ladder_strict']]}")
    save_json(out, METRICS / "p4_composite.json")


if __name__ == "__main__":
    main()
