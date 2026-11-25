# Data availability

All inputs are **public end-of-day files published by the National Stock Exchange of India (NSE)**.
No proprietary, paid or intraday data is used. Raw files are **not redistributed** in this repository
(`data/raw/` and `data/processed/` are git-ignored); they are re-downloaded from NSE's public archives by
`python -m impi.ingest.download` and rebuilt deterministically.

| Dataset | Used for | Public source (pattern from `config/data.yaml`) | Period used |
|---|---|---|---|
| Full bhavcopy with delivery (`sec_bhavdata_full`) | prices, volume, delivery % | `nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv` | 2019-09-25 → 2026-10-09 |
| Legacy equity bhavcopy | prices, volume (before 2019-10) | `nsearchives.nseindia.com/content/historical/EQUITIES/{yyyy}/{MON}/cm{dd}{MON}{yyyy}bhav.csv.zip` | 2014-01 → 2019-09 |
| Security-wise delivery position (MTO) | delivery quantity (before 2019-10) | `nsearchives.nseindia.com/archives/equities/mto/MTO_{ddmmyyyy}.DAT` | 2014-01 → 2019-09 |
| F&O bhavcopy (legacy, then UDiFF) | stock futures OI, basis, option OI, expiry dates | `.../DERIVATIVES/{yyyy}/{MON}/fo{dd}{MON}{yyyy}bhav.csv.zip`; `.../content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip` | 2014-01 → 2026-10-09 |
| Participant-wise open interest | FII / DII / Pro / Client positioning | `nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{ddmmyyyy}.csv` | 2014-01 → 2026-10-09 |
| F&O ban list | event exclusion | `nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_{ddmmyyyy}.csv` | 2014-01 → 2026-10-09 |
| Index close file (incl. India VIX) | Nifty 500 benchmark, regime | `nsearchives.nseindia.com/content/indices/ind_close_all_{ddmmyyyy}.csv` | 2014-01 → 2026-10-09 |
| VIX history, corporate actions, board meetings, symbol changes, index lists | VIX gaps, split/bonus adjustment, earnings exclusion, symbol mapping, sectors | NSE website APIs / archive CSVs listed in `config/data.yaml` | as needed |

**Point in time.** Each file is stamped with its public release time (`data/release_lag.yaml`); a feature
for day T uses only files released before the T decision point, and trades enter at the T+1 open.

**Re-downloading.** About 17,000 requests at ≥ 1 s spacing (≈ 5 hours). The downloader caches files,
resumes after interruptions and logs every attempt to `data/raw/_meta/download_log.csv`. NSE occasionally
re-publishes or moves files; the QC report (`results/metrics/qc_report.json`) lists the dates, rows and
checks of the dataset used in the paper (3,161 trading days, 5,836,673 daily equity rows) so a rebuilt
dataset can be compared. Small differences can arise if NSE revises archived files.

**Derived results.** All study outputs (`results/metrics/`, locked) and post-hoc outputs (`results/posthoc/`)
are included in the repository, so every table and figure can be regenerated without the raw data by
`scripts/posthoc/s08_paper_assets.py`.

Use of NSE data is subject to NSE's terms of use; the authors claim no rights over it.
