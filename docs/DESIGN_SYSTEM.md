# Holdfast design system ("Lens")

One question drives every screen: **is this move real?** Price says a stock went up; delivery says
whether buyers kept the shares. The UI leads with that verdict in plain words, then lets anyone who wants
more open the detail.

## Principles
1. **Verdict first, numbers second.** Every stock shows a sentence ("Buyers are holding") before a number.
2. **Plain words, tap for more.** Every metric has a `?` explainer (glossary in `app/static/ds/lens.js`).
3. **Simple / Pro.** One switch in the nav. Simple = verdict cards, line chart, fewer columns. Pro = the full
   terminal: candles, tools, IMPI model, every column. Stored per browser (`dl-mode`).
4. **Never colour alone.** Up/down always carry ▲ ▼; verdicts always carry a word.
5. **Honest numbers.** p-values and "not proven" stay visible; nothing is presented as a prediction it isn't.

## Tokens (`app/static/ds/lens.css`)
| Token | Value | Use |
|---|---|---|
| `--paper` / `--canvas` | #FFF / #F5F5F7 (dark #161617 / #000) | page, sections, panels (apple.com) |
| `--ink` → `--ink-4` | #1D1D1F → #86868B | text levels (apple.com) |
| `--gold*` (Apple blue) | #0071E3 / dark #2997FF | the only accent: strong delivery, links, buttons, focus |
| `--up` / `--down` | #0B8A5A / #CC3B2F | price direction only, always with ▲ ▼ |
| `--slate` | #6B7385 | weak delivery (mostly same-day trading) |

**Type:** Apple system type — SF Pro Display (≥ 20px) and SF Pro Text via `-apple-system`; Helvetica Neue / Segoe UI fallback elsewhere ·
numbers in SF Pro with tabular figures. Semibold (600) headlines, Apple tracking (17px text at −0.022em). Base size 17px; tap targets ≥ 44px.

**Motion:** ease-out `cubic-bezier(.16,1,.3,1)`; 160 / 320 / 640 ms. Cross-page View Transitions (nav stays,
content glides; the stock symbol morphs from Scanner/Home into the Terminal). Numbers roll, charts draw in,
lists reorder with FLIP. All motion is off under `prefers-reduced-motion`.

## Verdict language
| Score band | Pill | Sentence |
|---|---|---|
| High (≥ 70) | **Strong** (marigold) | Buyers are holding more than usual |
| Neutral | **Normal** (grey) | Delivery is about usual |
| Low (< 30) | **Weak** (slate) | Mostly same-day trading |

## Files
- `app/static/ds/lens.css` – tokens + components (nav, buttons, segmented control, cards, pills, tables, sheets)
- `app/static/ds/lens.js` – shared shell: nav, Simple/Pro mode, search palette (`/`), explainers, reveal,
  number tweening, FLIP, formatting helpers
- Pages: `landing.html`, `terminal.html`, `scanner.html`, `market.html`, `research.html`, `about.html`
- Previous UI kept untouched in `app/static_legacy/` (not served).
