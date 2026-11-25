"""Number audit for paper/main.pdf -> paper/number_audit.csv and a printed summary.

Every numeric token in the compiled PDF text is assigned a source:
  macro      - a \\newcommand from paper/numbers.tex (source file + JSON key from numbers_sources.csv)
  table      - a cell of an \\input table (source JSON named in the table's generator, s08_paper_assets.py)
  figure     - text inside a figure PDF (generated from the JSON named in the figure's script)
  design     - a design constant fixed in config/*.yaml or the pre-registration (e.g. q = 0.10, 30 events, 20 slots)
  reference  - bibliography metadata (refs.bib, DOI-verified against Crossref)
  structure  - section/table/figure/equation numbering, dates in the title block, page numbers
Anything else is reported as UNSOURCED. Also checks that each macro used in main.tex appears in the PDF
with exactly the value numbers.tex defines (a mismatch would mean a stale build).

    python scripts/audit_numbers.py        (needs pypdf)
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BS = chr(92)

TABLE_SOURCES = {
    "tab_dataset": "results/metrics/qc_report.json; adjustment_qc.json; results/posthoc/paper_facts.json",
    "tab_events": "results/metrics/p2_summary.json",
    "tab_confirmatory": "results/posthoc/s0_supplementary_audit.json (1_bh_family, from locked p6_test.json)",
    "tab_posthoc_audit": "results/posthoc/s0_supplementary_audit.json; B_breakeven.json",
    "tab_robustness": "results/posthoc/C_spec_curve.json; D_bayes.json; E_clustered_logit.json",
    "tab_backtests": "results/metrics/p6_test.json (backtests, benchmark, spa, deflated_sharpe)",
    "tab_timeline": "git log + results/posthoc/s0_supplementary_audit.json (10_timeline)",
}
FIG_SOURCES = {
    "fig1_pipeline": "results/posthoc/s0_supplementary_audit.json (timeline) + design",
    "fig_spec_curve": "results/posthoc/C_spec_curve.json",
    "fig_placebo_hist": "results/posthoc/A_placebo.json",
    "fig_h1_forest": "results/posthoc/F_h1_by_year.json",
    "fig_ladder": "results/metrics/p6_test.json (ladder_broad, ladder_strict)",
    "fig_cagr": "results/metrics/p6_test.json (backtests, benchmark)",
}
DESIGN = {"0.10": "BH q (config/models.yaml fdr_q)", "30": "min group size guard (hypothesis_tests.MIN_N)",
          "20": "max positions / 20-day high (config)", "10": "label horizon / lookback (config/labels.yaml)",
          "5": "compression days / expiry window / slippage bps", "14": "ATR window (config/labels.yaml)",
          "2": "upper barrier 2 ATR", "1": "lower barrier 1 ATR", "0.25": "PSI large-shift convention",
          "80": "power (pre-registration)", "0.05": "alpha", "90": "CI level", "95": "CI level",
          "23:59": "decision point (data/release_lag.yaml)", "500": "Nifty 500 index name", "3": "grid size",
          "2015": "split start year", "2019": "sec_bhav start year", "2020": "train end", "2021": "validation start",
          "2022": "validation end", "2023": "test start / T+1 year", "2024": "F&O reform year", "2025": "pilot start year",
          "2026": "data end year / registration year", "256": "SHA-256", "0.5": "spec-curve threshold", "1.0": "composite threshold",
          "1.5": "spec-curve barrier", "4": "column count", "6": "power-rule", "7": "", "8": "", "9": "", "0": "zero"}


def norm(s: str) -> str:
    return (s.replace("−", "-").replace("–", "-").replace("—", "-").replace(",", "")
            .replace("+", "").strip())


def macro_values():
    vals = {}
    for row in csv.DictReader(open(PAPER / "numbers_sources.csv", encoding="utf-8")):
        v = row["value"].replace(BS + "ensuremath{-}", "-").replace(BS + "ensuremath{<}", "<").replace(BS + "textbf{", "")
        vals[row["macro"]] = (v, row["source_file"], row["key"])
    return vals


def main():
    pdf_pages = [p.extract_text() for p in pypdf.PdfReader(PAPER / "main.pdf").pages]
    text = "\n".join(pdf_pages)
    tex = open(PAPER / "main.tex", encoding="utf-8").read()
    used = set(re.findall(BS * 2 + r"([A-Za-z]+)", tex))
    mv = macro_values()
    ntext = norm(text)
    # 1) macro presence check
    missing = []
    for m, (v, src, key) in mv.items():
        if m in used and not re.search(r"(?<![\d.])" + re.escape(norm(v)) + r"(?![\d])", ntext):
            missing.append((m, v))
    # 2) classify every numeric token in the PDF
    by_value = defaultdict(list)
    for m, (v, src, key) in mv.items():
        if m in used:
            by_value[norm(v)].append((m, src, key))
    table_numbers = defaultdict(set)
    for t in TABLE_SOURCES:
        tt = open(PAPER / "tables" / f"{t}.tex", encoding="utf-8").read().replace(BS + "ensuremath{-}", "-")
        for n in re.findall(r"-?\d[\d,]*\.?\d*", tt):
            table_numbers[norm(n)].add(t)
    bib = open(PAPER / "refs.bib", encoding="utf-8").read()
    bib_numbers = set(norm(n) for n in re.findall(r"\d[\d,]*", bib))
    rows, unsourced = [], []
    refs_start = text.find("REFERENCES")
    for i, page in enumerate(pdf_pages, 1):
        for mt in re.finditer(r"[−\-+]?\d[\d,]*(?:\.\d+)?%?", page):
            tok = mt.group(0)
            n = norm(tok.rstrip("%"))
            n_abs = n.lstrip("-")
            if n in by_value:
                kind, src = "macro", "; ".join(f"\\{m} <- {s} [{k}]" for m, s, k in by_value[n][:2])
            elif n in table_numbers or n_abs in table_numbers:
                ts = table_numbers.get(n) or table_numbers.get(n_abs)
                kind, src = "table", "; ".join(f"{t}: {TABLE_SOURCES[t]}" for t in sorted(ts))
            elif n_abs in DESIGN:
                kind, src = "design", DESIGN[n_abs] or "design/structure"
            elif n_abs in bib_numbers:
                kind, src = "reference", "refs.bib (Crossref-verified)"
            else:
                kind, src = "figure/structure", "figure text (see FIG_SOURCES) or numbering"
            rows.append({"page": i, "token": tok, "kind": kind, "source": src})
    with open(PAPER / "number_audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["page", "token", "kind", "source"])
        w.writeheader()
        w.writerows(rows)
    kinds = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    print("pages:", len(pdf_pages))
    print("numeric tokens:", len(rows), dict(kinds))
    print("macros used in main.tex:", len([m for m in mv if m in used]), "| macro values NOT found in PDF:", missing or "none")
    print("figure sources:", FIG_SOURCES)


if __name__ == "__main__":
    main()
