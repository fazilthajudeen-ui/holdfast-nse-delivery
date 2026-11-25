"""Lens Assistant: answers plain-English questions from the project's own data.

No external AI service is called. Every sentence is built from the cached NSE data, the
Conviction Scores, the buyer-zone rule and the backtest results, so nothing is invented.

answer(store, question, current_symbol, detail_fn) -> {"answer", "load", "suggest"}
"""
from __future__ import annotations

import re

NOTE = "_Research tool, not investment advice._"
SUGGEST_STOCK = ["Summarize this stock", "IMPI model view", "Why this score?", "Is there a buyer zone?", "Past sharp rises"]
SUGGEST_MARKET = ["Top high-conviction stocks", "Today's breakouts", "Market positioning", "Rising on low delivery", "Does the IMPI model work?"]


def _pct(v, d=1):
    if v is None:
        return "–"
    s = "+" if v > 0 else "−" if v < 0 else ""
    return f"{s}{abs(v):.{d}f}%"


def _pts(v):
    if v is None:
        return "–"
    s = "+" if v > 0 else "−" if v < 0 else ""
    return f"{s}{abs(v):.1f} pts"


def _date(s):
    import datetime as dt
    return dt.date.fromisoformat(s).strftime("%d %b")


# ---------------------------------------------------------------- understanding
INTENTS = [
    ("help", r"\b(hi|hello|hey|help|what can you|who are you)\b"),
    ("impi_research", r"(impi|model|\bai\b|\bml\b|machine learning).*(work|accura|good|reliable|perform|trust)"),
    ("impi_market", r"(market position|positioning|\bfii|\bdii|institution|participant|futures position)"),
    ("impi_scan", r"(breakouts?\b|scanner|compression|\bvcp\b|setups?\b)"),
    ("impi_stock", r"(impi|model|probabilit|predict|chance|volatility|machine learning|\bai\b)"),
    ("compare", r"\b(compare|versus|vs\.?)\b"),
    ("watch", r"(low delivery|rising on low|trap|fake|weak rall|risky|speculat|same-day)"),
    ("zones_list", r"(who|which|list|stocks).*(buyer zone|accumulat)|(buyer zone|accumulat).*(stocks|now|today)"),
    ("top_low", r"(lowest|weakest|worst|bottom).*(conviction|score|stock)|low conviction stocks"),
    ("top_high", r"(top|best|strongest|highest|most).*(conviction|score|stock|delivery)|high conviction stocks"),
    ("definition", r"what (is|does) (a )?delivery|meaning of delivery|delivery % mean|what is delivery"),
    ("evidence", r"(backtest|evidence|proof|does\b.*\bwork|is it (true|valid|useful)|accura|result|reliable|significan|trust)"),
    ("method", r"(how (does|is|do)|method|formula|calculat|compute|work out).*(score|work|calculat|method)?|^method"),
    ("zone", r"(zone|regime|accumulat|buyers? (holding|zone))"),
    ("history", r"(past|history|previous|sharp rise|events?|rall(y|ies))"),
    ("explain", r"(why|explain|score|conviction|reason)"),
    ("summary", r"(summar|analy|overview|about|tell me|view on|outlook|how is|what about|status)"),
]


def detect_intent(q: str) -> str:
    ql = q.lower()
    for name, pat in INTENTS:
        if re.search(pat, ql):
            return name
    return "summary"


def find_symbols(s, q: str) -> list[str]:
    syms = set(s.today.index)
    found = []
    for tok in re.findall(r"[A-Za-z0-9&\-]{2,}", q):
        t = tok.upper()
        if t in syms and t not in found and t not in {"VS", "AND", "THE", "IS", "OF", "ON", "IN", "IT", "TOP", "LOW", "HIGH"}:
            found.append(t)
    if not found:                                        # company-name words, e.g. "reliance"
        words = {w for w in re.findall(r"[a-z]{4,}", q.lower())}
        for sym, name in s.names.items():
            first = re.findall(r"[a-z]{4,}", name.lower())[:1]
            if first and first[0] in words and sym in syms and first[0] not in {"india", "bank", "limited", "industries"}:
                found.append(sym)
    return found[:2]


# ---------------------------------------------------------------- answers
def _reading(m, ev):
    band, r20 = m.get("band"), m.get("ret_20") or 0
    if band == "High" and r20 > 0:
        return (f"Rising **with** buyers taking delivery above normal: the pattern our backtest found more durable "
                f"({ev['beat_high']:.0f}% of high-conviction rises beat the Nifty 500 over 60 days vs {ev['beat_low']:.0f}% for low).")
    if band == "High":
        return "Price is weak, but buyers are taking delivery well above normal: accumulation on weakness, worth watching."
    if band == "Low" and r20 > 10:
        return (f"Rising on **mostly same-day trading**: the pattern that faded more often in our backtest "
                f"({ev['rt_low']:.0f}% of low-delivery rises gave back half their gain vs {ev['rt_high']:.0f}% for high).")
    if band == "Low":
        return "Few shares are being taken home right now: little conviction from buyers."
    return "Delivery is close to its normal level: no strong signal either way."


def _evidence(s):
    up = s.stats["up"]
    g = {(r["group"], r["horizon"]): r for r in up["groups"]}
    rt = {r["group"]: r for r in up["round_trip"]["rows"]}
    hv = up["high_vs_low"]
    return {"n": up["n_events"], "d20": hv["20"]["diff"], "p20": hv["20"]["p"], "d60": hv["60"]["diff"], "p60": hv["60"]["p"],
            "beat_high": g[("High", 60)]["hit_rate"], "beat_low": g[("Low", 60)]["hit_rate"],
            "rt_high": rt["High"]["rate"], "rt_low": rt["Low"]["rate"]}


def summary(s, d):
    m, ev = d["method"], _evidence(s)
    if not m:
        return f"**{d['symbol']}** doesn't have enough history to be scored yet."
    lines = [f"**{d['symbol']}** ({d['name'] or 'NSE'}) closed at **₹{m['close']:,.2f}**, {_pct(m['chg_1d'], 2)} today and {_pct(m['ret_20'])} over 20 days."]
    lines.append(f"• Conviction **{min(99, int(m['score']))}/100, {m['band']}**: {m['deliv']:.1f}% of shares delivered vs its normal {m['normal']:.1f}% ({_pts(m['rel_5'])} on a 5-day average).")
    z = d["current_zone"]
    if z:
        kind = "Buyer zone" if z["kind"] == "accumulation" else "Low-delivery zone"
        lines.append(f"• **{kind}** active since {_date(z['start'])} ({z['days']} trading days).")
    else:
        lines.append("• No active buyer zone right now.")
    if d["events"]:
        e = d["events"][-1]
        out = f"{_pct(e['exc_60'])} vs Nifty after 60 days" if e["status"] == "complete" else "still in progress"
        lines.append(f"• Last sharp rise: {_date(e['date'])}, {_pct(e['move'], 0)} on **{e['group']}** conviction, {out}.")
    lines.append("")
    lines.append("**Reading:** " + _reading(m, ev))
    return "\n".join(lines + ["", NOTE])


def explain(s, d):
    m = d["method"]
    if not m:
        return f"**{d['symbol']}** doesn't have enough history to be scored yet."
    bm, hd = s.baseline_window, s.history_days
    sc = min(99, int(m["score"]))
    return "\n".join([
        f"Why **{d['symbol']}** scores **{sc}/100** ({m['band']}):",
        f"• **1. Delivery today:** {m['deliv']:.1f}% of traded shares were taken home by buyers.",
        f"• **2. Its normal:** {m['normal']:.1f}%, the median of its previous {bm} trading days.",
        f"• **3. Gap:** {_pts(m['rel_5'])} above/below normal on a 5-day average.",
        f"• **4. Rank:** that gap is higher than about {sc}% of this stock's own last {hd} days.",
        "",
        f"So the score compares {d['symbol']} only with **its own history**, never with other stocks. Below {s.band_low} is Low, {s.band_high}+ is High.",
    ])


def zone(s, d):
    rg, z = d["regime"], d["current_zone"]
    last = d["smooth"][-1]["rel"] if d["smooth"] else None
    head = (f"**{d['symbol']}** is in a **{'buyer zone' if z['kind'] == 'accumulation' else 'low-delivery zone'}** "
            f"since {_date(z['start'])} ({z['days']} trading days).") if z else f"**{d['symbol']}** has **no active zone** right now."
    recent = [f"• {_date(x['start'])} – {_date(x['end'])}: {'buyer' if x['kind'] == 'accumulation' else 'low-delivery'} zone, {x['days']} days"
              for x in d["zones"][-3:][::-1]]
    return "\n".join([head, f"Smoothed delivery is {_pts(last)} vs its normal.", ""] + (["Recent zones:"] + recent + [""] if recent else []) + [
        f"_A buyer zone starts when the 10-day smoothed delivery rises {rg['enter_pts']}+ pts above the stock's normal "
        f"and ends when it falls back under {rg['exit_pts']} pts. Past data only; descriptive, not a trading signal._"])


def history(s, d):
    if not d["events"]:
        return f"**{d['symbol']}** had no sharp rise (20%+ in 20 days) in the last year."
    rows = []
    for e in d["events"][::-1][:6]:
        out = f"{_pct(e['exc_60'])} vs Nifty after 60d" if e["status"] == "complete" else "in progress"
        rows.append(f"• {_date(e['date'])}: {_pct(e['move'], 0)} rise, delivery {e['deliv']:.0f}% vs normal {e['normal']:.0f}% → **{e['group']}** conviction, {out}")
    return "\n".join([f"Sharp rises in **{d['symbol']}** over the last year:"] + rows)


def top(s, kind, n=8):
    t = s.today.reset_index()
    if kind == "high":
        t = t.sort_values("score", ascending=False).head(n)
        head = "**Top high-conviction stocks today** (delivery far above their own normal):"
    elif kind == "low":
        t = t.sort_values("score").head(n)
        head = "**Lowest-conviction stocks today** (mostly same-day trading):"
    else:
        t = t[(t["band"] == "Low") & (t["ret_20"] > 0.10)].sort_values("ret_20", ascending=False).head(n)
        head = "**Rising on low delivery** (up 10%+ in 20 days, score below 30). These rises faded more often in our backtest:"
        if t.empty:
            return "No stock is rising 10%+ on low delivery today."
    rows = [f"• **{r.symbol}**: score {min(99, int(r.score))}, delivery {r.deliv_pct:.0f}% vs normal {r.baseline_60:.0f}%, 20d {_pct(r.ret_20 * 100)}"
            for r in t.itertuples()]
    return "\n".join([head] + rows + ["", "_Click a symbol in the watchlist to open its chart._"])


def zones_list(s, n=8):
    z = [(sym, v) for sym, v in s.zone_now.items() if v and v["kind"] == "accumulation" and sym in s.today.index]
    z.sort(key=lambda x: -x[1]["days"])
    if not z:
        return "No stock is in a buyer zone today."
    rows = [f"• **{sym}**: buyer zone for {v['days']} days (since {_date(v['start'])}), score {min(99, int(s.today.loc[sym, 'score']))}"
            for sym, v in z[:n]]
    return "\n".join([f"**{len(z)} stocks** are in a buyer zone today. Longest-running:"] + rows)


def compare(s, a, b):
    rows = []
    for sym in (a, b):
        r = s.today.loc[sym]
        zn = (s.zone_now.get(sym) or {}).get("kind")
        rows.append(f"• **{sym}**: score **{min(99, int(r.score))}** ({r.band}), delivery {r.deliv_pct:.0f}% vs normal {r.baseline_60:.0f}%, "
                    f"20d {_pct(r.ret_20 * 100)}, {'buyer zone' if zn == 'accumulation' else 'low-delivery zone' if zn == 'low' else 'no zone'}")
    sa, sb = s.today.loc[a, "score"], s.today.loc[b, "score"]
    lead = a if sa >= sb else b
    return "\n".join([f"**{a} vs {b}** on delivery conviction:"] + rows +
                     ["", f"**{lead}** currently shows the stronger buyer conviction relative to its own history.", "", NOTE])


def evidence(s):
    e = _evidence(s)
    sig60 = "significant" if e["p60"] < 0.05 else "not yet statistically significant"
    return "\n".join([
        f"We tested every sharp rise (20%+ in 20 days) of the past year: **{e['n']} rises** across Nifty 500 stocks.",
        f"• Over **20 days**, high-conviction rises earned **{_pts(e['d20'])}** more than low-conviction ones vs the Nifty 500 (p = {e['p20']:.2f}).",
        f"• **{e['beat_high']:.0f}% vs {e['beat_low']:.0f}%** beat the Nifty 500 over 60 days (high vs low).",
        f"• **{e['rt_high']:.0f}% vs {e['rt_low']:.0f}%** gave back half their rise within 60 days (high vs low).",
        f"• At **60 days** the gap is {_pts(e['d60'])}: {sig60} (p = {e['p60']:.2f}).",
        "", "One year of data is a pilot: clear over weeks, not yet proven over months.",
    ])


def method(s):
    return "\n".join([
        "The Conviction Score in four steps:",
        "• **1. Delivery:** share of today's traded shares actually taken home by buyers (from NSE's daily file).",
        f"• **2. Normal:** the stock's own median delivery over the previous {s.baseline_window} days.",
        "• **3. Gap:** how far above or below normal it has been, on a 5-day average.",
        f"• **4. Score:** that gap ranked 0–100 against the stock's own last {s.history_days} days. {s.band_high}+ is High, below {s.band_low} is Low.",
        "", "Only past data is used, so there is no look-ahead.",
    ])


DEFINITION = ("**Delivery %** is the share of a day's traded shares that buyers actually took into their demat accounts. "
              "The rest were bought and sold again the same day.\n\nHigh delivery means buyers are **holding**; "
              "low delivery means mostly **short-term trading**. NSE publishes it for every stock, every evening.")
HELP = ("I'm the **Holdfast assistant**. I answer from our own NSE data, so every number is real. Try:\n"
        "• \"Summarize CGCL\" or \"Why is the score 99?\"\n• \"Is there a buyer zone?\"\n• \"Compare TCS vs INFY\"\n"
        "• \"Top high-conviction stocks\" or \"Rising on low delivery\"\n• \"Does this method work?\"\n"
        "• IMPI (F&O stocks): \"IMPI model view\", \"Today's breakouts\", \"Market positioning\", \"Does the IMPI model work?\"")


# ---------------------------------------------------------------- IMPI (F&O model) answers
def _impi():
    try:
        from app.impi_api import STORE
        st = STORE.get()
        return st if st.ok else None
    except Exception:
        return None


IMPI_MISSING = "IMPI live data isn't built yet. Run `python -m impi.live` in the impi folder (it also runs every weekday evening)."


def _p(v):
    return "–" if v is None else f"{v * 100:.0f}%"


def _auc(m, k):
    v = (m or {}).get(k)
    return "–" if v is None else f"{v:.2f}"


def impi_stock(sym):
    st = _impi()
    if st is None:
        return IMPI_MISSING
    r = st.today.get(sym)
    if r is None:
        return f"**{sym}** is not in the F&O segment, so IMPI (which needs futures data) doesn't cover it."
    M = st.meta.get("models", {})
    def base(k):
        return (M.get(k) or {}).get("posthoc_base_rate")
    if r.get("long_buildup") == 1:
        bu = "long build-up"
    elif r.get("short_covering") == 1:
        bu = "short covering"
    elif r.get("short_buildup") == 1:
        bu = "short build-up"
    elif r.get("long_unwinding") == 1:
        bu = "long unwinding"
    else:
        bu = "no clear build-up"
    xs = r.get("xs_pct")
    lines = [f"**IMPI model view for {sym}** (F&O data as of {_date(st.meta['as_of'])}):",
             f"• **Breakout success** (+2 ATR before −1 ATR in 10 days): **{_p(r.get('p_breakout'))}** vs a base rate of {_p(base('label_10'))}.",
             f"• **Beats the median F&O stock** over 10 days: **{_p(r.get('p_outperform'))}**; rank **{'–' if xs is None else round(xs)}/100** today.",
             f"• **Volatility expansion** (10-day vol ×1.5): **{_p(r.get('p_vol_expansion'))}** vs a base rate of {_p(base('vol_exp'))}."]
    v4 = r.get("v4") or {}
    if v4.get("move_pct") is not None:
        lines.append(f"• **Delivery signals (v4)**: expected move size **{round(v4['move_pct'])}/100** today (high delivery preceded bigger moves "
                     f"in either direction in the 2023–26 check); delivery flow **{v4.get('flow') or '–'}** (informational — not supported as a "
                     f"direction signal)." + (" **Big move watch.**" if v4.get("move_watch") else ""))
    z, bbw, oi = r.get("deliv_z20"), r.get("bbw_pct252"), r.get("oi_chg20")
    zt = "–" if z is None else f"{z:+.2f}σ"
    bt = "–" if bbw is None else str(round(bbw))
    ot = "–" if oi is None else f"{oi * 100:+.1f}%"
    lines.append(f"• Delivery vs its own year: {zt} · Bollinger width percentile {bt} · futures OI over 20 days {ot} ({bu}).")
    ev = [e for e in r.get("events", []) if e != "all"]
    if ev:
        lines.append("• Today: **" + " and ".join("VCP breakout" if e == "strict" else "compression breakout" for e in ev) + "**.")
    drivers = (r.get("explain") or {}).get("label_xs10") or []
    if drivers:
        lines.append("• Biggest drivers of its rank today: " + ", ".join(
            f"{d['feature'].replace('_', ' ')} ({'+' if d['contrib'] > 0 else '−'})" for d in drivers[:3]) + ".")
    lines += ["", f"_Model quality (CV / post-hoc test AUC): rank {_auc(M.get('label_xs10'), 'cv_auc')} / {_auc(M.get('label_xs10'), 'posthoc_test_auc')}, "
                  f"volatility {_auc(M.get('vol_exp'), 'cv_auc')} / {_auc(M.get('vol_exp'), 'posthoc_test_auc')}. "
                  "Volatility is predictable; direction only slightly. Not investment advice._"]
    return "\n".join(lines)


def impi_market():
    st = _impi()
    if st is None:
        return IMPI_MISSING
    m = st.market.iloc[-1]
    def g(c):
        v = m.get(c)
        return None if v is None or v != v else float(v)
    lines = [f"**Market positioning** (NSE participant-wise futures OI, {_date(st.meta['as_of'])}):"]
    for p, name in [("fii", "FIIs"), ("dii", "DIIs"), ("pro", "Prop desks"), ("client", "Clients")]:
        a, b, ca = g(f"{p}_fut_idx_share"), g(f"{p}_fut_stk_share"), g(f"{p}_fut_idx_cot")
        at = "–" if a is None else f"{a * 100:+.0f}%"
        btx = "–" if b is None else f"{b * 100:+.0f}%"
        ct = "" if ca is None else f" ({ca:.0f}/100 of its 52-week range)"
        lines.append(f"• **{name}**: net {at} of index-futures OI{ct}, {btx} of stock-futures OI.")
    tr, vix = g("nifty_trend_50_200"), g("vix_pct252")
    trt = "–" if tr is None else f"{tr * 100:+.1f}%"
    vt = "–" if vix is None else str(round(vix))
    lines.append(f"• Nifty 50d vs 200d average: {trt} · India VIX percentile: {vt}.")
    lines += ["", "_These are market-wide totals; NSE does not publish positions per stock. FII index shorts are often hedges. See the Market page._"]
    return "\n".join(lines)


def impi_scan():
    st = _impi()
    if st is None:
        return IMPI_MISSING
    rows = list(st.today.values())
    ev = [r for r in rows if any(e in ("broad", "strict") for e in r.get("events", []))]
    lines = [f"**F&O breakouts on {_date(st.meta['as_of'])}:**"]
    if ev:
        for r in sorted(ev, key=lambda x: -(x.get("xs_pct") or 0))[:8]:
            kind = "VCP" if "strict" in r["events"] else "compression"
            lines.append(f"• **{r['symbol']}**: {kind} breakout · P(breakout) {_p(r.get('p_breakout'))} · rank {round(r.get('xs_pct') or 0)}/100")
    else:
        lines.append("• No compression or VCP breakouts in F&O stocks today.")
    top = sorted(rows, key=lambda x: -(x.get("xs_pct") or 0))[:5]
    lines += ["", "Highest model rank vs F&O stocks today: " + ", ".join(f"**{r['symbol']}**" for r in top) + ".",
              "", "_In the research, breakouts did not beat the base rate after costs. Open the Scanner for the full table. Not investment advice._"]
    return "\n".join(lines)


def impi_research():
    st = _impi()
    if st is None:
        return IMPI_MISSING
    rs, M, tr = st.research, st.meta.get("models", {}), st.meta.get("track", {})
    conf = rs.get("v1_confirmatory", [])
    sup = sum(1 for t in conf if t.get("supported"))
    v1 = rs.get("v1_ml_test") or {}
    xa = (v1.get("xgb") or {}).get("auc")
    ca = v1.get("composite_auc")
    ver = "/".join(sorted({(m or {}).get("version") or "v2" for m in M.values()})) or "v2"
    lines = ["**Does the IMPI model work? Honest answer:**",
             f"• The pre-registered study (test set 2023–2026, run once) supported **{sup} of {len(conf)}** hypotheses. "
             f"Its ML model lost to a simple rule (AUC {'–' if xa is None else f'{xa:.3f}'} vs {'–' if ca is None else f'{ca:.3f}'}).",
             f"• The models now in use ({ver}; CV → post-hoc test AUC): volatility expansion {_auc(M.get('vol_exp'), 'cv_auc')} → {_auc(M.get('vol_exp'), 'posthoc_test_auc')} (useful); "
             f"beating the median stock over 10 days {_auc(M.get('label_xs10'), 'cv_auc')} → {_auc(M.get('label_xs10'), 'posthoc_test_auc')} and over 20 days "
             f"{_auc(M.get('label_xs20'), 'cv_auc')} → {_auc(M.get('label_xs20'), 'posthoc_test_auc')} (small, consistent edge, not tradable after costs); "
             f"breakout success over 10 days {_auc(M.get('label_10'), 'cv_auc')} → {_auc(M.get('label_10'), 'posthoc_test_auc')} (weak, unstable) and over 20 days "
             f"{_auc(M.get('label_20'), 'cv_auc')} → {_auc(M.get('label_20'), 'posthoc_test_auc')} (no edge).",
             "• v3 added option-implied volatility from the F&O bhavcopy. Its 2023+ numbers are post-hoc; the confirmatory test is the "
             "pre-registered forward track record.",
             f"• Live track record since {tr.get('first_logged', '–')}: {tr.get('predictions', 0)} predictions logged before their outcomes; "
             "accuracy is shown once 50+ have matured (10 trading days).",
             "", "_So: good at forecasting volatility, weak at forecasting direction. Research tool, not investment advice. Details on the Research page._"]
    return "\n".join(lines)


def answer(s, q: str, current: str | None, detail_fn) -> dict:
    q = (q or "").strip()
    syms = find_symbols(s, q)
    intent = detect_intent(q) if q else "help"
    if intent == "compare" and len(syms) == 2:
        return {"answer": compare(s, *syms), "load": None, "suggest": SUGGEST_MARKET}
    market = {"top_high": lambda: top(s, "high"), "top_low": lambda: top(s, "low"), "watch": lambda: top(s, "watch"),
              "zones_list": lambda: zones_list(s), "evidence": lambda: evidence(s), "definition": lambda: DEFINITION,
              "help": lambda: HELP, "impi_market": impi_market, "impi_scan": impi_scan, "impi_research": impi_research}
    if intent == "impi_stock":
        sym = syms[0] if syms else (current.upper() if current else None)
        if not sym:
            return {"answer": "Which stock? Try \"IMPI model view for TCS\".", "load": None, "suggest": SUGGEST_MARKET}
        return {"answer": impi_stock(sym), "load": sym if syms and sym != (current or "").upper() else None,
                "suggest": SUGGEST_STOCK}
    if intent in market and not (intent == "watch" and syms):
        return {"answer": market[intent](), "load": None, "suggest": SUGGEST_MARKET if intent != "help" else SUGGEST_STOCK}
    if intent == "method" and not syms:
        return {"answer": method(s), "load": None, "suggest": SUGGEST_STOCK}
    sym = syms[0] if syms else (current.upper() if current else None)
    if not sym or sym not in s.today.index:
        return {"answer": HELP if not sym else f"I couldn't find **{sym}** among the {len(s.today)} stocks I track.",
                "load": None, "suggest": SUGGEST_MARKET}
    d = detail_fn(sym)
    fn = {"explain": explain, "zone": zone, "history": history, "method": explain}.get(intent, summary)
    return {"answer": fn(s, d), "load": sym if syms and sym != (current or "").upper() else None, "suggest": SUGGEST_STOCK}
