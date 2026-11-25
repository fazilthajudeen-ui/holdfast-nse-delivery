# Delivery-Based Movement Analysis on NSE: Pilot Summary

*Data 2025-07-28 → 2026-11-25 · analysis window from 2025-11-25 · generated 2026-11-25*

**Question.** When a stock jumps sharply, does it matter whether buyers actually *took delivery* of the shares?
Do moves backed by unusually high delivery hold their gains better over the next 1–3 months than moves on low delivery?

**Data.** NSE's free daily "full bhavcopy" gives, for every stock, the shares traded and the shares actually delivered.
487 liquid Nifty 500 stocks (≥ 200 trading days, median turnover ≥ ₹1 crore), prices adjusted for splits/bonuses.
Benchmark: Nifty 500 index.

**Method.** A *move* = the price rising ≥ 20% from its lowest close in the previous 20 trading days
(one event per stock per 40 days). For each move we compare delivery during the move with the stock's own
60-day normal (median, measured *before* the move), and split the 789 up-moves into three equal groups:
Low, Mid and High conviction. We then track what happened over the next 5, 20 and 60 trading days.

**Headline.** High-conviction up-moves beat low-conviction up-moves by 0.0 percentage points of excess return over 60 trading days (n = 210 vs 194, p = 0.987, 95% CI -4.8 to +4.7). The difference is not statistically significant at 5%.

By horizon (High − Low excess return): +5 days +1.3 pp (p = 0.015, significant); +20 days +2.5 pp (p = 0.039, significant); +60 days +0.0 pp (p = 0.987).

**Yes in the short term, not proven over 3 months:** high-delivery moves did significantly better over the following weeks; at 60 days the gap is larger but noisier and could be chance (p ≥ 0.05).

| Conviction group | Completed events | Avg. excess return, 60 days | % beating Nifty 500 | Gave back ≥ half the rise | Avg. worst drop in 60 days |
|---|---|---|---|---|---|
| Low | 194 | +7.7% | 59% | 57% | -17.5% |
| Mid | 213 | +6.4% | 56% | 51% | -16.9% |
| High | 210 | +7.7% | 58% | 45% | -16.5% |

Round-trip difference across groups: chi-square p = 0.051.
Robustness: re-running with move sizes 15%, 20%, 25% and windows
10, 20, 30 days, the High-minus-Low gap points the same way in
**5 of 8** alternative settings (0 of 9 significant at 5%).
Down-moves (secondary): High − Low excess return = -0.1 pp (p = 0.981).

**Example pair (picked at random from matched events):** NMDC rose 20% on 43% delivery (+3 pp vs its normal) and was -9% 60 days later; HONASA rose 20% on 19% delivery (-29 pp) and was +13%.

**Limitations.** One year is a pilot: few events and a single market regime, so results are indicative, not proof.
Delivery shows that shares were taken, not who took them or why (block deals, promoter buying and index rebalancing
can inflate it). Groups are formed with the full year's events (fine for description, not a trading rule).
The universe is today's Nifty 500 list (survivorship bias). No transaction costs; this is not a trading strategy.

*Research / education tool. Not investment advice.*
