# IMPI — Indian Market Positioning Intelligence (research system)

A reproducible research codebase that tests whether the **interaction** of positioning changes,
delivery behaviour and price compression carries incremental information about **breakout success**
and **volatility expansion** in Indian equities (NSE, 2015-06 → 2026-10).

It extends the team's *Delivery Lens* project (one folder up), which keeps running unchanged.
Research tool only — **not investment advice**.

Team 23 · Hindustan Institute of Technology & Science · B.Tech CSE (AIML) · Supervisor Dr. Jeyavim Sherin RC

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
