"""Daily delivery features per stock -> data/derived/features.parquet.

    deliv_pct        = DELIV_QTY / TTL_TRD_QNTY * 100
    baseline_60      = median deliv_pct over the previous 60 trading days (shifted 1 day)
    rel_deliv        = deliv_pct - baseline_60
    deliv_qty_ratio  = DELIV_QTY / median DELIV_QTY of the previous 60 days (shifted 1 day)
    rel_deliv_5      = 5-day mean of rel_deliv (input to the Conviction Score)
    ret_20           = 20-trading-day price change

Every rolling statistic on day t uses rows up to t only (baselines use up to t-1).

    python -m src.features
"""
from __future__ import annotations

import pandas as pd

from src.common import load_config, log, path


def add_features(panel: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    w = cfg["features"]["baseline_window"]
    mp = cfg["features"]["baseline_min_periods"]
    k = cfg["score"]["rel_avg_days"]
    pc = cfg["score"]["price_change_days"]

    df = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["deliv_pct"] = (df["deliv_qty"] / df["volume"] * 100).where(df["volume"] > 0).clip(0, 100)
    g = df.groupby("symbol", sort=False)

    def past_median(col):
        return g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=mp).median())

    df["baseline_60"] = past_median("deliv_pct")
    df["rel_deliv"] = df["deliv_pct"] - df["baseline_60"]
    df["deliv_qty_ratio"] = df["deliv_qty"] / past_median("deliv_qty")
    df["rel_deliv_5"] = g["rel_deliv"].transform(lambda s: s.rolling(k, min_periods=3).mean())
    df["ret_20"] = g["close"].transform(lambda s: s / s.shift(pc) - 1)
    return df


def main():
    panel = pd.read_parquet(path("clean"))
    df = add_features(panel)
    df.to_parquet(path("features"), index=False)
    last = df["date"].max()
    today = df[df["date"] == last]
    print("FEATURES CHECK")
    print(f"  rows {len(df):,}, symbols {df['symbol'].nunique()}, last date {last:%Y-%m-%d}")
    print(f"  baseline available: {df['baseline_60'].notna().mean() * 100:.1f}% of rows "
          f"(first ~{load_config()['features']['baseline_min_periods']} days per stock are warm-up)")
    print(f"  deliv_pct median {df['deliv_pct'].median():.1f}%, "
          f"IQR {df['deliv_pct'].quantile(.25):.1f}-{df['deliv_pct'].quantile(.75):.1f}%")
    print(f"  on {last:%Y-%m-%d}: {today['rel_deliv'].notna().sum()} stocks with rel_deliv")
    log(f"saved {path('features').name}")


if __name__ == "__main__":
    main()
