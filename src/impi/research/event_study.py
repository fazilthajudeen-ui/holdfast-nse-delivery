"""Event study: mean forward excess returns (vs Nifty 500) at 5/10/20/60 days after entry, by group,
with month-block bootstrap confidence intervals, plus success rates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.common import config
from impi.research.stats import month_block_bootstrap


def event_table(ev: pd.DataFrame, group_col: str, rng, n_boot: int | None = None) -> pd.DataFrame:
    n_boot = n_boot or int(config()["models"]["bootstrap"]["n"]) // 4
    rows = []
    for g, d in ev.groupby(group_col, observed=True):
        row = {"group": g, "n": len(d), "success_rate_10": float(d["label_10"].mean())}
        for h in config()["labels"]["forward_returns"]:
            col = f"fwd_ex_{h}"
            x = d[d[col].notna()]
            b = month_block_bootstrap(x, lambda z, c=col: float(z[c].mean()), n_boot, rng) if len(x) > 20 else \
                {"estimate": float(x[col].mean()) if len(x) else np.nan, "ci_low": np.nan, "ci_high": np.nan}
            row[f"mean_ex_{h}"] = b["estimate"]
            row[f"ci_low_{h}"] = b["ci_low"]
            row[f"ci_high_{h}"] = b["ci_high"]
            row[f"n_{h}"] = len(x)
        rows.append(row)
    return pd.DataFrame(rows)
