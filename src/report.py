"""Export slide-ready PNG charts and a one-page plain-English summary.

    python -m src.report      -> reports/charts/*.png, reports/summary.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import charts
from src.common import DISCLAIMER, load_config, log, path
from src.events import similar_pairs


def load_all():
    with open(path("stats"), encoding="utf-8") as f:
        stats = json.load(f)
    return stats, pd.read_parquet(path("events")), pd.read_parquet(path("features"))


def pick_example_pair(ev: pd.DataFrame, seed: int):
    pairs = similar_pairs(ev)
    if pairs.empty:
        return None
    row = pairs.sample(1, random_state=seed).iloc[0]
    return ev.loc[row["high_idx"]], ev.loc[row["low_idx"]]


def pick_example_stock(ev: pd.DataFrame) -> str:
    """Stock whose completed up-move had the highest delivery conviction (deterministic)."""
    done = ev[(ev["direction"] == "up") & (ev["status"] == "complete")]
    src = done if len(done) else ev
    return src.sort_values("event_rel_deliv", ascending=False).iloc[0]["symbol"]


def save(fig, name: str, width=1280, height=None):
    out = path("charts") / name
    try:
        fig.write_image(out, width=width, height=height or fig.layout.height, scale=2)
    except Exception as e:           # kaleido needs a local Chrome; keep going with HTML
        out = out.with_suffix(".html")
        fig.write_html(out, include_plotlyjs="cdn")
        log(f"PNG export failed ({e.__class__.__name__}); wrote {out.name} instead")
    return out


def fmt(x, spec="+.1f", suffix=""):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:{spec}}{suffix}"


def write_summary(stats: dict, pair, cfg: dict) -> str:
    h = stats["headline_horizon"]
    up = stats["up"]
    g = pd.DataFrame(up["groups"])
    g60 = g[g["horizon"] == h].set_index("group")
    rt = {r["group"]: r for r in up["round_trip"]["rows"]}
    c = up["high_vs_low"][str(h)]
    e = cfg["events"]
    rob = pd.DataFrame(stats["robustness"])
    down = stats["down"]["high_vs_low"][str(h)]

    if c["diff"] is None:
        verdict = "There are not yet enough completed events to answer the question."
    elif c["label"] != "ok":
        verdict = "Too few events for a firm answer (n < 30 in a group); read the numbers as indicative."
    elif c["p"] < 0.05 and c["diff"] > 0:
        verdict = "**Yes, in this pilot year:** moves on high delivery held their gains better, and the gap is statistically significant."
    elif c["diff"] > 0 and any(v["p"] is not None and v["p"] < 0.05 and v["diff"] > 0
                               for k, v in up["high_vs_low"].items() if k != str(h)):
        verdict = ("**Yes in the short term, not proven over 3 months:** high-delivery moves did "
                   "significantly better over the following weeks; at 60 days the gap is larger but noisier "
                   "and could be chance (p ≥ 0.05).")
    elif c["diff"] > 0:
        verdict ="**Leaning yes, but not proven:** high-delivery moves did better on average, but the gap could be chance (p ≥ 0.05)."
    else:
        verdict = "**No, not in this pilot year:** high-delivery moves did not hold their gains better than low-delivery moves."

    horizons = "; ".join(
        f"+{k} days {v['diff']:+.1f} pp (p = {v['p']:.3f}{', significant' if v['p'] < 0.05 else ''})"
        for k, v in up["high_vs_low"].items() if v["diff"] is not None)
    pair_line = ""
    if pair is not None:
        hi, lo = pair
        pair_line = (f"\n**Example pair (picked at random from matched events):** {hi['symbol']} rose "
                     f"{hi['move_pct']:.0f}% on {hi['event_deliv']:.0f}% delivery "
                     f"({hi['event_rel_deliv']:+.0f} pp vs its normal) and was {hi[f'ret_{h}']:+.0f}% "
                     f"{h} days later; {lo['symbol']} rose {lo['move_pct']:.0f}% on {lo['event_deliv']:.0f}% "
                     f"delivery ({lo['event_rel_deliv']:+.0f} pp) and was {lo[f'ret_{h}']:+.0f}%.\n")

    rows = "\n".join(
        f"| {grp} | {int(g60.loc[grp, 'n'])} | {fmt(g60.loc[grp, 'mean_excess'], '+.1f', '%')} "
        f"| {fmt(g60.loc[grp, 'hit_rate'], '.0f', '%')} | {fmt(rt[grp]['rate'], '.0f', '%')} "
        f"| {fmt(rt[grp]['mean_max_dd_60'], '.1f', '%')} |"
        for grp in ["Low", "Mid", "High"])

    md = f"""# Delivery-Based Movement Analysis on NSE: Pilot Summary

*Data {stats['data_start']} → {stats['data_end']} · analysis window from {stats['analysis_start']} · generated {stats['generated'][:10]}*

**Question.** When a stock jumps sharply, does it matter whether buyers actually *took delivery* of the shares?
Do moves backed by unusually high delivery hold their gains better over the next 1–3 months than moves on low delivery?

**Data.** NSE's free daily "full bhavcopy" gives, for every stock, the shares traded and the shares actually delivered.
{stats['n_stocks_universe']} liquid Nifty 500 stocks (≥ {cfg['universe']['min_trading_days']} trading days, median turnover ≥ ₹1 crore), prices adjusted for splits/bonuses.
Benchmark: {stats['benchmark']}.

**Method.** A *move* = the price rising ≥ {e['move_pct']:.0%} from its lowest close in the previous {e['window']} trading days
(one event per stock per {e['cooldown']} days). For each move we compare delivery during the move with the stock's own
60-day normal (median, measured *before* the move), and split the {up['n_events']} up-moves into three equal groups:
Low, Mid and High conviction. We then track what happened over the next 5, 20 and {h} trading days.

**Headline.** {stats['headline']}

By horizon (High − Low excess return): {horizons}.

{verdict}

| Conviction group | Completed events | Avg. excess return, {h} days | % beating Nifty 500 | Gave back ≥ half the rise | Avg. worst drop in {h} days |
|---|---|---|---|---|---|
{rows}

Round-trip difference across groups: chi-square p = {fmt(up['round_trip']['p_3x2'], '.3f')}.
Robustness: re-running with move sizes {', '.join(f'{int(m * 100)}%' for m in cfg['robustness']['move_pcts'])} and windows
{', '.join(str(w) for w in cfg['robustness']['windows'])} days, the High-minus-Low gap points the same way in
**{stats['robustness_holds']}** alternative settings ({int((rob['significant_5pct'] == True).sum())} of {len(rob)} significant at 5%).
Down-moves (secondary): High − Low excess return = {fmt(down['diff'], '+.1f', ' pp')} (p = {fmt(down['p'], '.3f')}).
{pair_line}
**Limitations.** One year is a pilot: few events and a single market regime, so results are indicative, not proof.
Delivery shows that shares were taken, not who took them or why (block deals, promoter buying and index rebalancing
can inflate it). Groups are formed with the full year's events (fine for description, not a trading rule).
The universe is today's Nifty 500 list (survivorship bias). No transaction costs; this is not a trading strategy.

*{DISCLAIMER}*
"""
    return md


def main():
    cfg = load_config()
    stats, ev, feats = load_all()
    h = stats["headline_horizon"]
    out = []
    out.append(save(charts.tercile_returns(stats, "up"), "01_tercile_returns.png"))
    out.append(save(charts.roundtrip_rate(stats, "up"), "02_roundtrip_rate.png", width=1000))

    pair = pick_example_pair(ev, cfg["stats"]["seed"])
    if pair is not None:
        hi, lo = pair
        paths = {}
        for e_, tag in [(hi, "High conviction"), (lo, "Low conviction")]:
            prices = feats.loc[feats["symbol"] == e_["symbol"], ["date", "close"]]
            paths[f"{e_['symbol']}: {tag} ({e_['event_deliv']:.0f}% delivery)"] = charts.event_path(prices, e_, post=h, pre=10)
        out.append(save(charts.event_pair(paths, f"Two similar {hi['move_pct']:.0f}% / {lo['move_pct']:.0f}% rallies, different delivery",
                                          reveal=True, horizon=h), "03_example_event_pair.png"))

    sym = pick_example_stock(ev)
    df = feats[feats["symbol"] == sym]
    out.append(save(charts.stock_chart(df, ev[ev["symbol"] == sym], f"{sym}: price vs delivery"),
                    "04_delivery_vs_price.png", height=720))
    out.append(save(charts.tercile_returns(stats, "down"), "05_down_moves_tercile_returns.png"))

    md = write_summary(stats, pair, cfg)
    (path("reports") / "summary.md").write_text(md, encoding="utf-8")
    for o in out:
        log(f"wrote {o.relative_to(path('reports').parent)}")
    log("wrote reports/summary.md")


if __name__ == "__main__":
    main()
