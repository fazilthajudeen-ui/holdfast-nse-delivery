"""Descriptive dataset facts used in the paper that are not stored in any results JSON (counts only, no
outcomes): distinct symbols in the equity store and F&O underlyings per day. Writes paper_facts.json.

    .venv/Scripts/python.exe scripts/posthoc/s09_paper_facts.py
"""
from __future__ import annotations

import pandas as pd

from _common import save
from impi.transforms import pit_store as pit


def main():
    eq = pit.read("equity_raw", columns=["date", "symbol"])
    fo = pit.read("fo_stock", columns=["date", "symbol"])
    per_day = fo.groupby("date")["symbol"].nunique()
    save({"task": "descriptive dataset facts (no outcomes)",
          "n_symbols_equity_store": int(eq["symbol"].astype(str).nunique()),
          "equity_first_date": str(pd.Timestamp(eq["date"].min()).date()), "equity_last_date": str(pd.Timestamp(eq["date"].max()).date()),
          "n_fo_underlyings_ever": int(fo["symbol"].astype(str).nunique()),
          "fo_underlyings_per_day_min": int(per_day.min()), "fo_underlyings_per_day_max": int(per_day.max())},
         "paper_facts.json")


if __name__ == "__main__":
    main()
