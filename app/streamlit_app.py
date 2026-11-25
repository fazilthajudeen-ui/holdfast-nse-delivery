"""Delivery-Based Movement Analysis on NSE — live dashboard.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import charts  # noqa: E402
from src.common import DISCLAIMER, load_config  # noqa: E402
from src.events import similar_pairs  # noqa: E402

st.set_page_config(page_title="Delivery Conviction · NSE", page_icon="📦", layout="wide")
CFG = load_config()
P = {k: ROOT / v for k, v in CFG["paths"].items()}
H = CFG["stats"]["headline_horizon"]

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  h1 {font-size: 2.4rem !important;}
  [data-testid="stMetricValue"] {font-size: 2.3rem;}
  [data-testid="stMetricLabel"] p {font-size: 1.05rem;}
  .headline {font-size: 1.7rem; line-height: 1.35; font-weight: 600; padding: 1.2rem 1.4rem;
             border-left: 6px solid #2a78d6; background: #eef4fc; border-radius: 6px; margin: .6rem 0 1rem;}
  .disclaimer {font-size: .95rem; color: #52514e; border-top: 1px solid #e1e0d9; padding-top: .5rem; margin-top: 1.5rem;}
  .pill {display:inline-block; padding: .15rem .7rem; border-radius: 999px; font-weight: 600; color: white;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_data(stamp: tuple):    # no leading underscore: Streamlit must hash it into the cache key
    stats = json.loads(P["stats"].read_text(encoding="utf-8"))
    ev = pd.read_parquet(P["events"])
    feats = pd.read_parquet(P["features"])
    scores = pd.read_parquet(P["scores"])
    rob = pd.read_parquet(P["events_all_params"]) if P["events_all_params"].exists() else pd.DataFrame()
    return stats, ev, feats, scores, rob


def data_stamp() -> tuple:
    """Cache key: every file the app reads plus the run log (written last by update_daily),
    so a page opened mid-rebuild cannot pin a half-updated snapshot."""
    files = [P["stats"], P["events"], P["features"], P["scores"], P["events_all_params"],
             P["logs"] / "run_history.csv"]
    return tuple((f.stat().st_mtime_ns, f.stat().st_size) if f.exists() else (0, 0) for f in files)


if not P["scores"].exists() or not P["stats"].exists():
    st.error("No derived data yet. Run the pipeline first (see README): "
             "`python -m src.download --months 14` then clean → features → events → stats → score.")
    st.stop()

STATS, EV, FEATS, SCORES, ROB = load_data(data_stamp())
AS_OF = SCORES["date"].max()
UPDATED = dt.datetime.fromtimestamp(P["scores"].stat().st_mtime)
TODAY = SCORES[SCORES["date"] == AS_OF].dropna(subset=["score"]).copy()
TODAY["ret_20_pct"] = TODAY["ret_20"] * 100


def footer():
    st.markdown(f"<div class='disclaimer'>⚠️ {DISCLAIMER} Data: NSE daily bhavcopy with delivery "
                f"(as of {AS_OF:%d %b %Y}). 1-year pilot study: indicative, not proof.</div>",
                unsafe_allow_html=True)


def pill(band: str | None) -> str:
    color = charts.BAND_COLORS.get(band or "", charts.MUTED)
    return f"<span class='pill' style='background:{color}'>{band or 'n/a'}</span>"


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("Do sharp moves on high delivery hold their gains?")
    st.caption(f"Data as of **{AS_OF:%A, %d %b %Y}** · last refreshed {UPDATED:%d %b %Y, %H:%M}")
    st.markdown(f"<div class='headline'>{STATS['headline']}</div>", unsafe_allow_html=True)
    hv = STATS["up"]["high_vs_low"]
    st.markdown("**By horizon:** " + " · ".join(
        f"+{k} days: {v['diff']:+.1f} pp (p = {v['p']:.3f}{' ✔ significant' if v['p'] < 0.05 else ''})"
        for k, v in hv.items() if v["diff"] is not None))

    up = STATS["up"]
    c = up["high_vs_low"][str(H)]
    rt = {r["group"]: r for r in up["round_trip"]["rows"]}
    k = st.columns(4)
    k[0].metric("Stocks tracked", f"{len(TODAY)}")
    k[1].metric("Sharp up-moves studied", f"{up['n_events']}", help=f"{up['n_complete_60']} with a full {H}-day outcome")
    if c["diff"] is not None:
        k[2].metric(f"High − Low ({H} days)", f"{c['diff']:+.1f} pp",
                    help=f"Difference in average excess return vs Nifty 500; p = {c['p']:.3f}")
    if rt["High"]["rate"] is not None and rt["Low"]["rate"] is not None:
        k[3].metric("Gave back half the rise", f"{rt['High']['rate']:.0f}% vs {rt['Low']['rate']:.0f}%",
                    help="High-conviction vs Low-conviction up-moves, within 60 days")

    st.subheader("How to read the Conviction Score")
    a, b, cc = st.columns(3)
    a.markdown(f"{pill('High')}  **70–100**: far more shares are being *taken home* than this stock's usual. "
               "Buyers are holding.", unsafe_allow_html=True)
    b.markdown(f"{pill('Neutral')}  **30–70**: delivery near its normal level.", unsafe_allow_html=True)
    cc.markdown(f"{pill('Low')}  **0–30**: mostly same-day trading; few shares change hands for real.",
                unsafe_allow_html=True)
    bands = TODAY["band"].value_counts()
    st.markdown(f"**Today:** {bands.get('High', 0)} stocks High · {bands.get('Neutral', 0)} Neutral · "
                f"{bands.get('Low', 0)} Low.")
    with st.expander("What is delivery %?"):
        st.write("Every day NSE reports, for every stock, how many traded shares were actually delivered into "
                 "buyers' demat accounts. A price rise on 15% delivery is mostly same-day speculation; a rise on "
                 "60% delivery means buyers are taking and keeping the shares. The score compares each stock with "
                 "its own history, because 'normal' delivery differs a lot between stocks.")
    footer()


def page_explorer():
    st.title("Stock Explorer")
    symbols = sorted(TODAY["symbol"].unique())
    default = symbols.index("RELIANCE") if "RELIANCE" in symbols else 0
    sym = st.selectbox("Search a symbol", symbols, index=default)
    row = TODAY[TODAY["symbol"] == sym]
    df = FEATS[FEATS["symbol"] == sym].sort_values("date")
    ev = EV[EV["symbol"] == sym]

    left, right = st.columns([1, 2.2])
    with left:
        r = row.iloc[0] if len(row) else None
        st.plotly_chart(charts.gauge(r["score"] if r is not None else None, r["band"] if r is not None else None),
                        width="stretch", config={"displayModeBar": False})
        if r is not None:
            m1, m2 = st.columns(2)
            m1.metric("Delivery today", f"{r['deliv_pct']:.0f}%", f"{r['rel_deliv']:+.1f} pp vs normal",
                      delta_color="off")
            m2.metric("20-day price change", f"{r['ret_20_pct']:+.1f}%")
            st.caption(f"Normal (60-day median) delivery: {r['baseline_60']:.0f}%")
    with right:
        n = st.radio("Period", ["3 months", "6 months", "All"], index=2, horizontal=True)
        days = {"3 months": 63, "6 months": 126, "All": len(df)}[n]
        view = df.tail(days)
        evv = ev[ev["event_date"] >= view["date"].min()]
        st.plotly_chart(charts.stock_chart(view, evv, sym, candles=days <= 130), width="stretch")
    if len(ev):
        st.subheader(f"Past sharp moves in {sym}")
        show = ev[["event_date", "direction", "move_pct", "event_deliv", "event_rel_deliv", "group",
                   f"ret_{H}", f"exc_{H}", "round_trip", "status"]].copy()
        show["event_date"] = show["event_date"].dt.date
        show["round_trip"] = show["round_trip"].map({1.0: "yes", 0.0: "no"})
        st.dataframe(show.rename(columns={
            "event_date": "Date", "direction": "Direction", "move_pct": "Move %", "event_deliv": "Delivery %",
            "event_rel_deliv": "vs normal (pp)", "group": "Conviction", f"ret_{H}": f"Return +{H}d %",
            f"exc_{H}": f"Excess +{H}d %", "round_trip": "Round-trip", "status": "Status"}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                           ["Move %", "Delivery %", "vs normal (pp)", f"Return +{H}d %", f"Excess +{H}d %"]})
    else:
        st.info("No sharp moves (≥ 20% in 20 days) for this stock in the analysis window.")
    footer()


def page_leaderboard():
    st.title("Today's Leaderboard")
    st.caption(f"Conviction Scores for {AS_OF:%d %b %Y} · {len(TODAY)} stocks")
    preset = st.radio("Show", ["All stocks", "Up > 10% in 20 days", "Down > 10% in 20 days"], horizontal=True)
    lo_, hi_ = float(np.floor(TODAY["ret_20_pct"].min())), float(np.ceil(TODAY["ret_20_pct"].max()))
    rng = {"All stocks": (lo_, hi_), "Up > 10% in 20 days": (10.0, hi_), "Down > 10% in 20 days": (lo_, -10.0)}[preset]
    rng = st.slider("20-day price change (%)", lo_, hi_, rng, step=1.0, key=f"ret20_{preset}")
    t = TODAY[TODAY["ret_20_pct"].between(*rng)]
    cols = {"symbol": "Symbol", "score": "Score", "deliv_pct": "Delivery %",
            "baseline_60": "Normal %", "ret_20_pct": "20-day change %"}
    csv_cols = {**cols, "band": "Band", "rel_deliv_5": "5-day vs normal (pp)", "close": "Close ₹"}
    cfg = {c: st.column_config.NumberColumn(format="%.1f") for c in
           ["Delivery %", "Normal %", "20-day change %"]}
    cfg["Score"] = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f", width="small")
    cfg["Symbol"] = st.column_config.TextColumn(width="medium")
    lo_band, hi_band = CFG["score"]["band_low"], CFG["score"]["band_high"]

    def board(frame, empty_msg):
        if frame.empty:
            st.info(empty_msg)
        else:
            st.dataframe(frame[list(cols)].rename(columns=cols), hide_index=True, width="stretch",
                         column_config=cfg, height=min(740, 38 + 35 * len(frame)))

    a, b = st.columns(2)
    with a:
        st.subheader("🔵 High conviction")
        st.caption(f"Top 20 with score ≥ {hi_band}")
        if preset == "Up > 10% in 20 days":
            st.caption("Moving up **with** buyers taking delivery")
        board(t[t["score"] >= hi_band].nlargest(20, "score"), "No High-conviction stocks match this filter.")
    with b:
        st.subheader("⚪ Low conviction")
        st.caption(f"Top 20 with score < {lo_band}")
        if preset == "Up > 10% in 20 days":
            st.caption("**Watch list:** moving up on mostly same-day trading")
        board(t[t["score"] < lo_band].nsmallest(20, "score"), "No Low-conviction stocks match this filter.")
    st.download_button("Download all filtered stocks (CSV)",
                       t.sort_values("score", ascending=False)[list(csv_cols)].rename(columns=csv_cols)
                       .to_csv(index=False),
                       file_name=f"conviction_{AS_OF:%Y-%m-%d}.csv", mime="text/csv")
    footer()


def page_history():
    st.title("Historical Analysis: 1-year pilot")
    direction = st.radio("Moves", ["Up-moves (main question)", "Down-moves (secondary)"], horizontal=True)
    d = "up" if direction.startswith("Up") else "down"
    s = STATS[d]
    st.caption(f"{s['n_events']} events in {s['n_stocks']} stocks · analysis window {STATS['analysis_start']} → "
               f"{STATS['data_end']} · benchmark: {STATS['benchmark']} · * = fewer than 30 events (insufficient data)")
    if d == "up":
        st.markdown(f"<div class='headline'>{STATS['headline']}</div>", unsafe_allow_html=True)
    st.plotly_chart(charts.tercile_returns(STATS, d), width="stretch")
    st.plotly_chart(charts.roundtrip_rate(STATS, d), width="stretch")

    st.subheader("High vs Low conviction, by horizon")
    hv = pd.DataFrame([{**v, "horizon": f"+{k} days"} for k, v in s["high_vs_low"].items()])
    st.dataframe(hv[["horizon", "n_high", "n_low", "mean_high", "mean_low", "diff", "ci_low", "ci_high", "p", "label"]]
                 .rename(columns={"horizon": "Horizon", "n_high": "n High", "n_low": "n Low", "mean_high": "High mean %",
                                  "mean_low": "Low mean %", "diff": "Difference (pp)", "ci_low": "95% CI low",
                                  "ci_high": "95% CI high", "p": "Welch p", "label": "Note"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                ["High mean %", "Low mean %", "Difference (pp)", "95% CI low", "95% CI high"]}
                 | {"Welch p": st.column_config.NumberColumn(format="%.3f")})

    if d == "up" and len(ROB):
        st.subheader(f"Robustness: does the answer survive other definitions of a 'sharp move'?")
        st.caption(f"Same direction as the main result in {STATS['robustness_holds']} alternative settings.")
        rb = ROB.copy()
        rb["setting"] = rb.apply(lambda r: f"≥{r.move_pct}% in {r.window} days" + (" (main)" if r.is_baseline else ""), axis=1)
        st.dataframe(rb[["setting", "n_events", "n_high", "n_low", f"diff_exc_{H}", "p_value", "rt_rate_high",
                         "rt_rate_low", "same_direction", "label"]]
                     .rename(columns={"setting": "Move definition", "n_events": "Events", "n_high": "n High",
                                      "n_low": "n Low", f"diff_exc_{H}": f"High − Low, {H}d (pp)", "p_value": "p",
                                      "rt_rate_high": "Round-trip High %", "rt_rate_low": "Round-trip Low %",
                                      "same_direction": "Same direction?", "label": "Note"}),
                     hide_index=True, width="stretch",
                     column_config={f"High − Low, {H}d (pp)": st.column_config.NumberColumn(format="%+.2f"),
                                    "p": st.column_config.NumberColumn(format="%.3f"),
                                    "Round-trip High %": st.column_config.NumberColumn(format="%.0f"),
                                    "Round-trip Low %": st.column_config.NumberColumn(format="%.0f")})

    st.subheader("All events")
    e = EV[EV["direction"] == d].copy()
    e["round_trip"] = e["round_trip"].map({1.0: "yes", 0.0: "no"})
    for c_ in ("start_date", "event_date"):
        e[c_] = e[c_].dt.date
    cols = ["symbol", "start_date", "event_date", "move_pct", "event_deliv", "baseline_start", "event_rel_deliv",
            "group", "exc_5", "exc_20", f"exc_{H}", "round_trip", "max_dd_60", "status"]
    st.dataframe(e[cols], hide_index=True, width="stretch", height=420,
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                ["move_pct", "event_deliv", "baseline_start", "event_rel_deliv", "exc_5", "exc_20",
                                 f"exc_{H}", "max_dd_60"]})
    st.markdown("**Limitations.** One year is a pilot: few events, one market regime. Delivery shows shares were "
                "taken, not who took them or why (block deals, promoter buying, index rebalancing can inflate it). "
                "Groups use the whole year's events. Universe = today's Nifty 500 (survivorship bias). "
                "No transaction costs; not a trading strategy.")
    footer()


def page_demo():
    st.title("Guess the Stock")
    st.markdown("Two real stocks. Both jumped by a similar amount. **Which one held its gains better "
                f"over the next {H} trading days?**")
    pairs = similar_pairs(EV, horizon=H)
    if pairs.empty:
        st.info("Not enough completed events yet to build pairs.")
        return
    ss = st.session_state
    new_pair = st.button("🔀 New pair", type="primary")
    if "pair" not in ss or new_pair:
        i = int(np.random.default_rng().integers(len(pairs)))
        ss.pair, ss.flip, ss.revealed, ss.guess = i, bool(np.random.default_rng().integers(2)), False, None
    pr = pairs.iloc[ss.pair % len(pairs)]
    hi, lo = EV.loc[pr["high_idx"]], EV.loc[pr["low_idx"]]
    order = [(lo, "Low"), (hi, "High")] if ss.flip else [(hi, "High"), (lo, "Low")]

    paths = {}
    for (e, grp), name in zip(order, ["Stock A", "Stock B"]):
        prices = FEATS.loc[FEATS["symbol"] == e["symbol"], ["date", "close"]]
        label = (f"{name}: {e['symbol']} ({grp} conviction: delivery {e['event_rel_deliv']:+.0f} pp vs its normal)"
                 if ss.revealed else f"{name}: rose {e['move_pct']:.0f}%")
        paths[label] = charts.event_path(prices, e, post=H, pre=10)
    st.plotly_chart(charts.event_pair(paths, reveal=ss.revealed, horizon=H), width="stretch")

    if not ss.revealed:
        g = st.radio("Your guess:", ["Stock A", "Stock B"], horizontal=True, index=None)
        if st.button("Reveal", disabled=g is None):
            ss.guess, ss.revealed = g, True
            st.rerun()
    else:
        rets = {"Stock A": order[0][0][f"ret_{H}"], "Stock B": order[1][0][f"ret_{H}"]}
        winner = max(rets, key=rets.get)
        high_name = "Stock A" if order[0][1] == "High" else "Stock B"
        (st.success if ss.guess == winner else st.error)(
            f"You picked {ss.guess}. {winner} held up better ({rets[winner]:+.0f}% vs "
            f"{min(rets.values()):+.0f}% after {H} days).")
        low_name = "Stock B" if high_name == "Stock A" else "Stock A"
        st.markdown(
            f"The **high-conviction** stock was **{high_name} ({hi['symbol']})**: during its rise buyers took delivery "
            f"of {hi['event_deliv']:.0f}% of shares, **{hi['event_rel_deliv']:+.0f} pp** vs its own normal of "
            f"{hi['baseline_start']:.0f}%.  \n"
            f"**{low_name} ({lo['symbol']})** saw {lo['event_deliv']:.0f}% delivery, **{lo['event_rel_deliv']:+.0f} pp** "
            f"vs its normal of {lo['baseline_start']:.0f}%. Conviction compares each stock with *itself*.")
        share = (pairs["high_ret"] > pairs["low_ret"]).mean() * 100
        st.caption(f"Across all {len(pairs)} matched pairs, the high-conviction stock held up better in "
                   f"{share:.0f}% of cases. Pairs are drawn at random from completed events, never hand-picked.")
    footer()


nav = st.navigation([
    st.Page(page_overview, title="Overview", icon="🏠", default=True),
    st.Page(page_explorer, title="Stock Explorer", icon="🔎", url_path="explorer"),
    st.Page(page_leaderboard, title="Today's Leaderboard", icon="🏆", url_path="leaderboard"),
    st.Page(page_history, title="Historical Analysis", icon="📊", url_path="history"),
    st.Page(page_demo, title="Demo: Guess the Stock", icon="🎯", url_path="demo"),
])
with st.sidebar:
    st.markdown(f"**Data as of**  \n{AS_OF:%d %b %Y}")
    st.caption("Delivery-Based Movement Analysis on NSE")
    st.caption(DISCLAIMER)
nav.run()
