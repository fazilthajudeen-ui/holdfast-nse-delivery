"""Daily Conviction Score for every stock -> data/derived/scores.parquet.

Score(t) = percentile rank (0-100) of rel_deliv_5(t) (5-day mean of delivery % above the
stock's own 60-day baseline) within that stock's own last 250 trading days (up to and
including t — no future data). Bands: 0-30 Low, 30-70 Neutral, 70-100 High.

    python -m src.score
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.common import load_config, log, path


def rolling_percentile(x: np.ndarray, window: int, min_history: int) -> np.ndarray:
    """Percentile rank of x[t] among x[t-window+1 .. t] (NaNs ignored); mid-rank for ties."""
    n = len(x)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    pad = np.concatenate([np.full(window - 1, np.nan), x])
    win = sliding_window_view(pad, window)               # row t = history ending at t
    cur = x[:, None]
    valid = ~np.isnan(win)
    cnt = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        below = (win < cur).sum(axis=1)
        equal = (win == cur).sum(axis=1)
        pct = (below + 0.5 * equal) / cnt * 100
    ok = (cnt >= min_history) & ~np.isnan(x)
    out[ok] = pct[ok]
    return out


def band(score: pd.Series, low: float, high: float) -> pd.Series:
    b = pd.Series(pd.NA, index=score.index, dtype="object")
    b[score < low] = "Low"
    b[(score >= low) & (score < high)] = "Neutral"
    b[score >= high] = "High"
    return b


def compute_scores(features: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    s = cfg["score"]
    df = features.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["score"] = np.nan
    for _, idx in df.groupby("symbol", sort=False).indices.items():
        df.loc[idx, "score"] = rolling_percentile(df["rel_deliv_5"].to_numpy()[idx],
                                                  s["history_days"], s["min_history"])
    df["band"] = band(df["score"], s["band_low"], s["band_high"])
    cols = ["date", "symbol", "close", "volume", "turnover_lacs", "deliv_pct", "baseline_60",
            "rel_deliv", "rel_deliv_5", "deliv_qty_ratio", "score", "band", "ret_20"]
    return df[cols]


def main():
    feats = pd.read_parquet(path("features"))
    sc = compute_scores(feats)
    sc.to_parquet(path("scores"), index=False)
    last = sc["date"].max()
    today = sc[sc["date"] == last].dropna(subset=["score"])
    print(f"CONVICTION SCORES for {last:%Y-%m-%d}: {len(today)} stocks scored")
    print("  bands:", today["band"].value_counts().to_dict())
    cols = ["symbol", "score", "deliv_pct", "baseline_60", "ret_20"]
    fmt = today.assign(ret_20=today["ret_20"] * 100)
    with pd.option_context("display.float_format", "{:.1f}".format):
        print("  top 5 High:\n" + fmt.nlargest(5, "score")[cols].to_string(index=False))
        print("  top 5 Low:\n" + fmt.nsmallest(5, "score")[cols].to_string(index=False))
    log(f"saved {path('scores').name}")


if __name__ == "__main__":
    main()
