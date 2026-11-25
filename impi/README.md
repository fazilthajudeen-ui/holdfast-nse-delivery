# IMPI — Indian Market Positioning Intelligence (research system)

A reproducible research codebase that tests whether the **interaction** of positioning changes,
delivery behaviour and price compression carries incremental information about **breakout success**
and **volatility expansion** in Indian equities (NSE, 2015-06 → 2026-10).

It extends the team's *Delivery Lens* project (one folder up), which keeps running unchanged.
Research tool only — **not investment advice**.


## Results at a glance (data 2014-01-01 → 2026-10-09; test set 2023+, evaluated once)

| | Finding |
|---|---|
| Data | 3,161 trading days from official NSE files; pre-2019 delivery rebuilt from legacy bhavcopy + MTO, matching `sec_bhavdata_full` exactly on 12 overlap dates; 1,442 split/bonus/demerger adjustments |
| Events | 4,876 compression breakouts (BROAD), 11,327 twenty-day breakouts, only 82 VCP (STRICT) events in Tier 1 |
| H1 delivery | no effect (test 24.8% vs 24.9%, p = 0.99) |
| H2 divergence | −2.8 pts in the predicted direction, not significant (q = 0.29) |
| H3 FII/DII regime | no effect |
| H4 ML vs rule | **loss**: XGBoost AUC 0.460 vs composite 0.526 (DeLong q = 0.024); walk-forward 0.502 |
| Economics | breakout strategies lose after costs; SPA p = 0.77, Reality Check p = 0.89 vs Nifty 500 |

Full tables: `results/summary.md` · paper draft: `paper/IMPI_paper_draft.md` · every decision: `decisions.md`.

## What is in here

| Path | Contents |
|---|---|
| `config/` | every number the pipeline uses: data, universe, features, labels, splits, costs, models |
| `data/release_lag.yaml` | when each dataset becomes public (enforces the no-look-ahead rule G1) |
| `src/impi/ingest/` | NSE downloaders and parsers (equity bhavcopy, MTO delivery, F&O legacy + UDiFF, participant OI, ban list, indices, corporate actions, results dates) |
| `src/impi/transforms/` | point-in-time Parquet store, split/bonus adjustment of prices **and** quantities |
| `src/impi/features/` | delivery, volume, compression, VCP detector, positioning, divergence, regime; `build.py` = events + labels |
| `src/impi/labels/` | triple-barrier and volatility-expansion labels, event triggers and exclusions |
| `src/impi/research/` | H1–H3 tests, event studies, DeLong, block bootstrap, power |
| `src/impi/models/` | rule-based composite, XGBoost / RF / logistic, isotonic calibration, SHAP |
| `src/impi/backtest/` | Indian cost model, portfolio executor, metrics, deflated Sharpe, SPA / Reality Check |
| `src/impi/validation/` | purged splits and CV, Benjamini–Hochberg, robustness |
| `src/impi/pipeline/` | one runner per phase (P0–P8) |
| `tests/` | unit tests on synthetic data + `integration/make_synthetic.py` (full rehearsal) |
| `results/` | `results_log.csv` (every variant tried), `metrics/`, `figures/`, `summary.md` |
| `pre_registered_hypotheses.md` | hypotheses, committed to git BEFORE any test was run |
| `decisions.md` | every ambiguity and how it was resolved |

## Guardrails (from the spec) and how the code enforces them

| | Rule | Enforcement |
|---|---|---|
| G1 | no look-ahead | release times in `data/release_lag.yaml`; features end at T, baselines at T−1; VCP pivots used only once confirmed; earnings exclusion uses intimation timestamps; P1 truncation-invariance test on real data |
| G2 | no survivorship | universe rebuilt daily from that day's bhavcopies (delisted names included); F&O membership from that day's F&O bhavcopy |
| G3 | chronological only | contiguous time splits; purged K-fold CV with embargo; no shuffling |
| G4 | test touched once | `results/test_lock.json`; P6 refuses a second run |
| G5 | log every variant | `results/results_log.csv` (search configs, backtests, grid points, tests) |
| G6 | adjust prices AND volumes | `transforms/adjust.py`; OI in shares too |
| G7 | seeds + config hash | one seed in `config/models.yaml`; every result row stores config hash + git commit |

## Run it (Windows, from this folder)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest
.venv/Scripts/python -m impi.pipeline.run_all --to p2
```

Then commit `pre_registered_hypotheses.md` and continue:

```bash
.venv/Scripts/python -m impi.pipeline.run_all --from p3 --to p5
.venv/Scripts/python -m impi.pipeline.p6_test
.venv/Scripts/python -m impi.pipeline.run_all --from p7
```

The first download is about 17,000 polite requests (~5 hours); files are cached and the downloader
resumes where it stopped.

## Reproducing the paper

Environment: Python 3.12.10 on Windows 11 x64. `requirements.txt` pins the research packages to the
versions in the `.venv` that produced the results; `requirements-lock.txt` is the full `pip freeze`.
Licence: `LICENSE` (MIT, code only). Data: `DATA_AVAILABILITY.md` (public NSE files, not redistributed).

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt
.venv/Scripts/python -m pip install -e .
```

**1. Pre-registered study (already run once; P6 refuses to run again).** Runtimes measured on the
authors' laptop.

| Step | Command | Runtime |
|---|---|---|
| Download + point-in-time store + QC + G1 check + features/events/freeze | `.venv/Scripts/python -m impi.pipeline.run_all --to p2` | ~5 h download, ~20 min build |
| Pre-registration | commit `pre_registered_hypotheses.md` (done: `15103f1`) | – |
| P3 hypotheses, P4 composite/backtests, P5 ML search + freeze (train/validation only) | `.venv/Scripts/python -m impi.pipeline.run_all --from p3 --to p5` | ~30 min |
| P6 one-time test evaluation (writes `results/test_lock.json`) | `.venv/Scripts/python -m impi.pipeline.p6_test` | ~7 min |
| P7 robustness, P8 report | `.venv/Scripts/python -m impi.pipeline.run_all --from p7` | ~10 min |

**2. Post-hoc robustness suite (after the test unlock; every output is labelled POST-HOC).** Run from
`scripts/posthoc/` in this order. None of these scripts writes to a locked file; step 0 records SHA-256
checksums of the locked files before, and the last step verifies them after.

| Step | Script | Output (`results/posthoc/`) | Runtime |
|---|---|---|---|
| 0 | `s00_checksums.py before` | `checksums_before.json` | seconds |
| supplementary audit (BH ranks, TOST, MDE, expiry, PSI, Brier, labels, timeline) | `s0_supplementary_audit.py` | `s0_supplementary_audit.json` | ~2 min |
| A placebo, 1,000 within-day delivery shuffles | `s01_placebo.py 1000` | `A_placebo.json`, `paper/figs/fig_placebo_hist.pdf` | ~60–70 min |
| B cost break-even CIs | `s02_breakeven.py` | `B_breakeven.json` | ~5–8 min |
| D Bayes factors | `s03_bayes.py` | `D_bayes.json` | ~5–10 min |
| E cluster-robust eq. (3) | `s04_clustered_logit.py` | `E_clustered_logit.json` | < 1 min |
| F H1 by year | `s05_forest_by_year.py` | `F_h1_by_year.json`, `paper/figs/fig_h1_forest.pdf` | ~3–5 min |
| G breakout vs base, barrier ties | `s06_breakout_ties.py` | `G_breakout_ties.json` | < 1 min |
| C specification curve (72 specs) | `s07_spec_curve.py` | `C_spec_curve.json`, `paper/figs/fig_spec_curve.pdf` | < 1 min |
| H paper tables and figures (from JSON only) | `s08_paper_assets.py` | `paper/tables/*.tex`, `paper/figs/*.pdf` | < 1 min |
| last | `s00_checksums.py after` | `checksums_after.json` (exit code 1 if anything changed) | seconds |

```bash
cd scripts/posthoc
../../.venv/Scripts/python s00_checksums.py before
../../.venv/Scripts/python s0_supplementary_audit.py
../../.venv/Scripts/python s01_placebo.py 1000
../../.venv/Scripts/python s02_breakeven.py
../../.venv/Scripts/python s03_bayes.py
../../.venv/Scripts/python s04_clustered_logit.py
../../.venv/Scripts/python s05_forest_by_year.py
../../.venv/Scripts/python s06_breakout_ties.py
../../.venv/Scripts/python s07_spec_curve.py
../../.venv/Scripts/python s08_paper_assets.py
../../.venv/Scripts/python s00_checksums.py after
```

LaTeX: tables use `booktabs` (`\usepackage{booktabs}`); `paper/tables/all_tables.tex` inputs them all.
Seeds are fixed in each script, so results should reproduce exactly with the same package versions.

## After the study: model v2, live mode and the terminal

* **Model v2 (post-hoc):** `python -m impi.v2.panel` then `python -m impi.v2.train` (and `python -m impi.v2.ranker`).
  Developed on 2015–2022 only with walk-forward CV; 2023+ numbers are post-hoc (see `decisions.md` D-044).
* **Live mode:** `python -m impi.live` rebuilds into `data/live/`, scores every F&O stock with the frozen v2
  models, exports files for the web app and appends the forward track record (`data/track/`). It runs every
  weekday evening from `../scripts/run_daily.bat` (Task Scheduler "NSE Delivery Update", 19:30).
* **Terminal:** from the project root, `python -m app.server` → http://localhost:8000
  - `/terminal` — every stock's chart; the new **IMPI model** panel (probabilities vs base rates, percentile,
    delivery/compression/positioning snapshot, top drivers) and IMPI breakout markers (toggle "IMPI")
  - `/scanner` — all F&O stocks today, sortable, filter by breakout type and sector
  - `/market` — FII/DII/Pro/Client futures positioning, Nifty 500, India VIX, regime tiles
  - `/research` — pre-registered results, v2 model quality, live track record, figures
  - Ask Lens assistant: "IMPI model view", "Today's breakouts", "Market positioning", "Does the IMPI model work?"
