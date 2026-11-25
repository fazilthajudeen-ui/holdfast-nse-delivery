# PROJECT BRIEF: Delivery Lens (Delivery-Based Movement Analysis on NSE)

You are being given the complete description of an existing, working project. Use it to understand,
explain, present, maintain or extend the project. Every number below is real and comes from the
project's own data. Do not invent numbers; if a figure is needed that is not here, compute it from
the data or say it is unknown.

---

## 1. Who, what and why

- **Project title:** Delivery-Based Movement Analysis on NSE (product name: **Delivery Lens**).
- **Type:** research + education project with a live product. **Not investment advice.**

**The idea in one paragraph.** Every evening the National Stock Exchange of India (NSE) publishes,
for every stock, the **delivery %**: the share of the day's traded shares that buyers actually took
into their demat accounts. The rest were bought and sold again the same day (intraday trading). A
price rise on low delivery (e.g. 15%) is mostly same-day speculation; a rise on high delivery (e.g.
60%) means buyers are taking and holding the shares. A normal price chart shows neither.

**Core research question.** Do sharp price rises backed by unusually high delivery hold their gains
better over the next 1 to 3 months than rises on low delivery?

**Objectives.**
1. Build a clean, reproducible dataset from free official NSE files.
2. Test the question properly with statistics (a 1-year pilot study).
3. Turn the answer into a live tool that updates every trading evening and shows a Conviction Score
   for every stock.

**Tagline:** "Don't just watch the price. Watch the delivery."

---

## 2. Data

- **Source file (one per trading day):**
  `https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv`
  Columns: SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE,
  CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER.
- **Benchmark:** the Nifty 500 closing value from NSE's daily
  `content/indices/ind_close_all_DDMMYYYY.csv` file (fallback: median return of the universe).
- **Universe:** today's Nifty 500 constituents list (`content/indices/ind_nifty500list.csv`,
  saved with a date each day), EQ series only, at least 200 trading days of data and median daily
  turnover of at least ₹1 crore.
- **Fallback source** for testing only: a GitHub mirror (`chartiny/nse-sec-bhavdata-full`) that covers
  up to about Dec 2025 (`--source github`).
- **Polite downloading:** browser User-Agent, cookie warm-up on nseindia.com, at least 1 second
  between requests, retries with exponential backoff, every file cached locally and never downloaded
  twice. NSE blocks most cloud servers, so downloads run on the laptop.
- **Current data (8 Oct 2026):** 297 trading days from 28 Jul 2025 to 8 Oct 2026; 483 stocks scored;
  about 9 lakh+ raw rows parsed; median delivery across all stocks and days is 48%.

### Data problems found and fixed (a key part of the work)
1. **Leading spaces** in column names and values: stripped. **"-"** in delivery columns: treated as missing.
2. **Holiday pages:** NSE returns 404 or an HTML page on holidays; files are validated to start with
   `SYMBOL`.
3. **Stale holiday files:** on some holidays NSE returns HTTP 200 with the *previous day's* file
   (e.g. 1 May 2026 contained 30 Apr data). The downloader checks the date written inside each file
   and rejects mismatches (13 found in the first download). Known holidays are remembered in
   `data/logs/no_data_dates.txt` (18 dates so far) so they are never requested again.
4. **Weekend special sessions** (Union Budget, Sunday 1 Feb 2026) are fetched via
   `download.extra_dates` in `config.yaml`. Safety net: if more than 5% of stocks look like they
   "split" on the same day, it is treated as a missing session, not corporate actions.
5. **PREV_CLOSE is not split-adjusted** in this file on ex-dates (e.g. Adani Power 1:5 split,
   22 Sep 2025: PREV_CLOSE 709.40, open 148.20). Corporate actions are therefore detected from the
   **opening gap**: if OPEN / previous CLOSE is below 0.72 or above 1.5 and within a few % of a standard
   split/bonus ratio (1/2, 1/3, 1/4, 1/5, 1/10, 1/20, 2/5, 2/3, 3/4, 3/5, 2, 5, 10), the factor is
   snapped to that ratio; extreme gaps (< 0.5 or > 2) that match no ratio (e.g. the Vedanta demerger,
   30 Apr 2026) use the actual ratio. All earlier prices are multiplied by the factor and quantities
   divided by it. Every adjustment is logged to `data/logs/corporate_actions.csv` (26 so far,
   including HDFC Bank 1:1 bonus, MCX, CAMS and Angel One splits).
6. **Series hops** (EQ → BE → EQ): BE/BZ days are kept for price continuity, but their delivery is
   blanked, because BE/T2T stocks are delivery-only by rule and would fake 100% delivery.

---

## 3. Method (all thresholds live in `config.yaml`)

**Per stock, per day (no look-ahead: day t uses data up to t only):**
- `deliv_pct = DELIV_QTY / TTL_TRD_QNTY × 100`
- `baseline_60` = median deliv_pct of the **previous** 60 trading days (shifted by 1 day, min 40 days)
- `rel_deliv = deliv_pct − baseline_60` (how far above/below its own normal)
- `deliv_qty_ratio = DELIV_QTY / previous-60-day median DELIV_QTY`
- `rel_deliv_5` = 5-day average of rel_deliv; `ret_20` = 20-day price change

**Key idea:** every stock is judged against **its own** normal, never a fixed cut-off. 40% delivery
is weak for a stock that normally sees 60% and strong for one that normally sees 25%.

**Sharp-rise events:** the close is at least **20%** above the most recent lowest close of the
previous **20** trading days. Cool-down: one event per stock per direction per **40** days. Events
must fall in the most recent 12 months (the extra ~2 months of data are warm-up for the baseline).
Down-moves (falls of 20%+) are analysed as a secondary study.

**Event scoring:** `event_deliv` = Σ DELIV_QTY / Σ TTL_TRD_QNTY from the day after the low to the event
day; `event_rel_deliv` = event_deliv − baseline_60 on the low day. Events are split into **terciles**
(Low / Mid / High conviction) by event_rel_deliv.

**Outcomes:** returns after +5, +20 and +60 trading days, raw and **excess vs the Nifty 500**;
**round-trip** = gave back at least half of the rise within 60 days; maximum drawdown within 60 days.
Events too recent for a horizon are marked incomplete and excluded from that horizon.

**Statistics:** High vs Low difference in mean excess return, **Welch's t-test**, **bootstrap 95% CI**
(10,000 resamples), **chi-square** on round-trip rates, groups with n < 30 flagged "insufficient
data". **Robustness grid:** move size 15 / 20 / 25% × window 10 / 20 / 30 days.

**Conviction Score (live, daily, every stock):** the percentile rank (0–100) of today's
`rel_deliv_5` within the stock's **own last 250 trading days** (min 60). Bands: below 30 = **Low**,
30–70 = Neutral, 70+ = **High**. Displayed capped at 99 (a "100" looks suspicious to viewers).

**Buyer zones (chart regime, descriptive only, not backtested):** trailing 10-day EMA of rel_deliv.
A **buyer zone** (accumulation, shaded blue) starts when it rises to +5 pts or more and ends when it
falls below +2 pts (hysteresis stops flicker). A **low-delivery zone** (shaded orange) is the mirror
image (−5 / −2). Trailing data only. On 8 Oct 2026: 95 stocks in a buyer zone, 81 in a low-delivery
zone, 310 in neither.

---

## 4. Results

The results rebuild every evening, so the numbers move as new rises complete. Always quote the date.

**Pilot snapshot used in the slides (data to 28 Sep 2026):** 656 sharp rises in 359 stocks.
- High − Low excess return: **+1.3 pts at 5 days (p = 0.03)**, **+2.3 pts at 20 days (p = 0.03)**,
  +3.4 pts at 60 days (p = 0.11, not significant).
- Beat the Nifty 500 over 60 days: **61% (High) vs 52% (Low)**.
- Gave back half the rise within 60 days: **48% (High) vs 59% (Low)**.
- Same direction in 9 of 9 robustness settings. Down-moves: no difference (p = 0.89).

**Live numbers (data to 8 Oct 2026):** 679 sharp rises in 360 stocks.
- High − Low at 20 days: **+2.4 pts (p = 0.04, significant)**. 20-day mean excess: Low +1.1%,
  Mid +0.8%, High +3.5%.
- At 60 days: +1.7 pts (p = 0.49, **not** significant).
- Beat the Nifty 500 over 60 days: 61% vs 57%. Gave back half: 45% vs 55%.
- Robustness: same direction in 8 of 8 alternative settings. Down-moves: 271 events.

**Honest conclusion:** high-delivery rises hold up better **over the following weeks** (statistically
significant at 20 days in every run); over 3 months the effect is **promising but not proven** with
one year of data. Presenting this honestly is deliberate.

**Limitations:** one year is a pilot (one market regime); delivery shows shares were taken, not who
took them or why (block deals, promoter buying, index rebalancing can raise it); terciles use the
whole year's events (fine for description, not a trading rule); the universe is today's Nifty 500
list (survivorship bias); no transaction costs; not a trading strategy.

---

## 5. The product

### 5.1 Daily automation
- `python -m src.update_daily` downloads any missing days, rebuilds every stage (≈3 minutes) and logs
  the run to `data/logs/run_history.csv`. Safe to run any number of times. If today's file is not out
  yet (NSE publishes ~6–7 PM IST) it exits cleanly. It looks back 7 days to retry failed days.
- Full rebuild instead of row patching, because a split on a new day back-adjusts that stock's
  whole history.
- **Windows Task Scheduler** task "NSE Delivery Update" runs it Mon–Fri at 19:30, set to run on
  battery and catch up if the laptop was asleep (`scripts/schedule_daily.bat`; cron helper in
  `scripts/schedule_daily.sh`). Optional `--push` commits and pushes data for a cloud dashboard.

### 5.2 Website (`python -m app.server` → http://localhost:8000)
FastAPI backend reading the Parquet/JSON outputs (auto-reloads when they change, no-cache headers).
- **`/` Landing page (Apple-style):** frosted sticky nav; hero "Watch the delivery." with a real
  terminal screenshot; "Half of every trade never leaves the day." with an animated 48/52 split bar;
  "One score. Four honest steps." (Delivery → Normal → Gap → Score); "The proof" with count-up
  numbers read live from the API; closing tagline and footer. Scroll-reveal motion, respects
  reduced-motion.
- **`/about`:** project, team, supervisor, department, institution, data, tech stack, disclaimer.
- **`/terminal`:** the trading terminal (below).

### 5.3 Trading terminal (TradingView-style, light and dark)
- **Charts:** TradingView's open-source **Lightweight Charts v5** (bundled offline). Candles; sharp-rise
  arrows coloured by conviction (blue High, grey Mid, orange Low); buyer/low-delivery zone shading;
  a second pane with delivery % bars (blue above normal, grey below), the 10-day smoothed delivery
  line and the 60-day normal (orange); faint symbol watermark.
- **Controls:** Daily / Weekly candles (weekly delivery is share-weighted), 1M / 3M / 6M / 1Y / All
  ranges (set by bar index so they work at any size), SMA 20 / SMA 50 / Zones toggles.
- **Drawing tools:** crosshair, trend line, horizontal line, measure (% change and bar count), reset
  view, clear drawings, save chart image, fullscreen.
- **Watchlist:** tabs High / Low / Rising · low (up 10%+ in 20 days on low delivery) / All; each row
  has a 20-day sparkline, last price, 1-day change, a score pill shaded by strength and a blue dot if
  the stock is in a buyer zone. Search by symbol or company name.
- **Top bar:** NIFTY 500 level and change, search, data date, theme toggle, shortcuts, home.
- **Right panel:** animated semicircle score gauge; "How it's calculated" (the 4 steps with this
  stock's real numbers); Buyer zone (current state, recent zones, explanation); Sharp rises in this
  stock (with 60-day outcome); Backtest evidence (bars and key facts).
- **Design:** Inter for text, JetBrains Mono for numbers (both bundled); light and dark themes with
  smooth cross-fade; subtle motion (fades, count-ups, gauge sweep, price flash, skeleton loading).
- **Keyboard:** `/` search, `↑↓` stocks, `D`/`W` interval, `1`–`5` range, `T` theme, `A` assistant,
  `F` fullscreen, `?` help, `Esc` cancel. The URL keeps stock, interval and range
  (e.g. `/terminal#CGCL?i=W&r=12`).

### 5.4 Lens Assistant (chatbot)
A chat panel ("Ask Lens", key `A`) that answers plain-English questions **from the project's own
data**. It is **rule-based, not a large language model**, and calls no external AI service, so every
number it gives is real. It understands: summaries ("summarize TCS", "tell me about reliance"),
"why this score?", buyer zones, past sharp rises, comparisons ("compare TCS vs INFY"), top
high/low-conviction lists, rising on low delivery, who is in a buyer zone, what delivery % means,
how the score works, and whether the method works (backtest). Mentioning a stock opens its chart.
Answers end with "Research tool, not investment advice." where relevant. Code: `app/assistant.py`.

### 5.5 API
`GET /api/meta` (as-of date, counts, NIFTY 500, method settings, backtest evidence),
`GET /api/watchlist` (all scored stocks with sparkline and zone),
`GET /api/stock/{SYMBOL}` (bars, delivery, normal, smoothed line, zones, events, method numbers),
`POST /api/ask` `{q, symbol}` → `{answer, load, suggest}`.

### 5.6 Also included
- Backup Streamlit dashboard: `streamlit run app/streamlit_app.py`.
- `python -m src.report` writes slide-ready PNG charts to `reports/charts/` and a one-page
  `reports/summary.md`.
- Presentation: 6 slides (title page image + Introduction & Problem, Our Solution, Technical Approach,
  Results & Validation, Impact & Future Scope), teal/amber design, speaker notes on every slide, saved
  as `23_Delivery-Based Movement Analysis on NSE.pptx`. Its results slide uses the 28 Sep snapshot.

---

## 6. Tech stack
Python 3.12 · requests · pandas · NumPy · pyarrow (Parquet) · SciPy · PyYAML · FastAPI + uvicorn ·
TradingView Lightweight Charts (JS) · Plotly + kaleido (report charts) · Streamlit (backup UI) ·
pytest · Windows Task Scheduler.

## 7. Repository layout
```
delivery-movement-nse/
  config.yaml              all thresholds, paths, source, regime settings
  requirements.txt         pinned versions
  README.md
  src/  download.py  clean.py  features.py  events.py  stats.py  score.py  regime.py
        report.py  charts.py  update_daily.py  common.py
  app/  server.py (FastAPI)  assistant.py  streamlit_app.py (backup)
        static/  landing.html  about.html  terminal.html  site.css  img/  vendor/ (charts lib, fonts)
  data/ raw/ (daily CSVs + index files)  universe/  clean/panel.parquet  clean/benchmark.parquet
        derived/ features.parquet events.parquet scores.parquet stats.json robustness.parquet
        logs/ (download log, corporate actions, holidays, run history, data-quality report)
  reports/ summary.md  charts/*.png
  scripts/ run_daily.bat  schedule_daily.bat  schedule_daily.sh
  tests/ test_pipeline.py  test_assistant.py   (34 tests, all passing)
```

## 8. Commands
```
pip install -r requirements.txt
python -m src.download --months 14 --source nse          # first time, ~10-15 min
python -m src.clean && python -m src.features && python -m src.events && python -m src.stats && python -m src.score && python -m src.report
python -m src.update_daily           # daily update (also scheduled 7:30 PM Mon-Fri)
python -m src.update_daily --force   # rebuild without new data
python -m app.server                 # website + terminal on http://localhost:8000
python -m pytest                     # tests
```

## 9. Quality and validation
- 34 automated tests: CSV parsing quirks, holiday and stale-file detection, split/bonus adjustment
  (including unadjusted PREV_CLOSE and a crash that must not be treated as a split), missing-session
  guard, series hops, event detection and cool-down, outcomes and incomplete flags, terciles,
  **no look-ahead** (baselines, scores and zones unchanged when future rows are added), regime
  hysteresis, and assistant question understanding.
- A full rebuild from an empty `data/` folder reproduced byte-identical results.
- Every stage prints a data-quality check; 10 random events are printed for manual chart checks.

## 10. Rules when working on or presenting this project
- Use only real numbers from the data and always say which date they are as of.
- State the 60-day result honestly as not yet proven; the 20-day result is the solid finding.
- Never present the Conviction Score or zones as buy/sell signals. It is a research and education tool.
- Never call the Lens Assistant an LLM; it is a rule-based assistant over the project's data.
- Keep the no-look-ahead rule in any new feature.
