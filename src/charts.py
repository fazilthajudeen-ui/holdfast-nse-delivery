"""Plotly figures shared by the dashboard and the PNG report (one visual system).

Palette: one accent hue (blue). Ordered groups (Low/Mid/High conviction) use one-hue
light->dark steps; the two-stock comparison uses categorical slots 1-2 (blue, orange).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ACCENT = "#2a78d6"
ORANGE = "#eb6834"
GROUP_COLORS = {"Low": "#86b6ef", "Mid": "#3987e5", "High": "#184f95"}
BAND_COLORS = {"Low": "#86b6ef", "Neutral": "#3987e5", "High": "#184f95"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _style(fig: go.Figure, title: str | None = None, height: int = 460, big: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, title=dict(text=title, x=0, font=dict(size=22 if big else 18, color=INK)),
        font=dict(family=FONT, size=16 if big else 13, color=INK2),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, margin=dict(l=60, r=30, t=70 if title else 30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hoverlabel=dict(font_family=FONT, bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


# ---------------------------------------------------------------- historical analysis
def tercile_returns(stats: dict, direction: str = "up") -> go.Figure:
    """Mean forward excess return by tercile for each horizon, 95% CI error bars, n labels."""
    g = pd.DataFrame(stats[direction]["groups"])
    fig = go.Figure()
    for grp in ["Low", "Mid", "High"]:
        x = g[g["group"] == grp]
        y = x["mean_excess"].astype(float)
        fig.add_bar(
            name=f"{grp} conviction", x=[f"+{h} days" for h in x["horizon"]], y=y,
            marker=dict(color=GROUP_COLORS[grp], line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", symmetric=False, color=INK2, thickness=1.5, width=6,
                         array=(x["ci_high"].astype(float) - y).clip(lower=0),
                         arrayminus=(y - x["ci_low"].astype(float)).clip(lower=0)),
            customdata=np.c_[x["n"], x["ci_low"], x["ci_high"]],
            hovertemplate=(f"<b>{grp}</b> %{{x}}<br>mean excess %{{y:+.2f}}%<br>"
                           "95% CI %{customdata[1]:+.2f} to %{customdata[2]:+.2f}<br>n = %{customdata[0]}<extra></extra>"),
        )
    # counts go under each horizon (Low / Mid / High), clear of the error bars
    ticks = {}
    for h in sorted(g["horizon"].unique()):
        ns = [int(g[(g["group"] == k) & (g["horizon"] == h)]["n"].iloc[0]) for k in ["Low", "Mid", "High"]]
        ticks[f"+{h} days"] = (f"<b>+{h} days</b><br><span style='font-size:13px'>n = "
                               + " / ".join(f"{n}{'*' if n < 30 else ''}" for n in ns) + "</span>")
    fig.update_xaxes(tickvals=list(ticks), ticktext=list(ticks.values()))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(title="Excess return vs Nifty 500 (%)", ticksuffix="%")
    what = "up-moves" if direction == "up" else "down-moves"
    return _style(fig, f"After sharp {what}: forward excess return by delivery conviction")


def roundtrip_rate(stats: dict, direction: str = "up") -> go.Figure:
    rows = pd.DataFrame(stats[direction]["round_trip"]["rows"])
    fig = go.Figure(go.Bar(
        x=rows["group"] + " conviction", y=rows["rate"].astype(float),
        marker=dict(color=[GROUP_COLORS[g] for g in rows["group"]]),
        text=[f"{r:.0f}%<br><span style='font-size:13px'>n={n}</span>" if pd.notna(r) else "–"
              for r, n in zip(rows["rate"], rows["n"])],
        textposition="outside", cliponaxis=False, textfont=dict(size=18, color=INK),
        hovertemplate="%{x}<br>round-trip rate %{y:.1f}%<extra></extra>"))
    p = stats[direction]["round_trip"]["p_3x2"]
    sub = f"chi-square p = {p:.3f}" if p is not None else ""
    word = "gave back ≥50% of the rise" if direction == "up" else "recovered ≥50% of the fall"
    fig.update_yaxes(title=f"% of events that {word}", ticksuffix="%", rangemode="tozero")
    fig.update_layout(showlegend=False, bargap=0.45)
    return _style(fig, f"Round-trips within 60 days <span style='font-size:15px;color:{MUTED}'>{sub}</span>")


# ---------------------------------------------------------------- one stock
def stock_chart(df: pd.DataFrame, events: pd.DataFrame | None = None, symbol: str = "",
                candles: bool = True, height: int = 640) -> go.Figure:
    """Price (top) and delivery % bars with the 60-day baseline (bottom). Two panels, one x-axis."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.05,
                        subplot_titles=("Price (split/bonus-adjusted, ₹)", "Delivery % of traded shares"))
    if candles and {"open", "high", "low"} <= set(df.columns):
        fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"],
                                     close=df["close"], name="Price", increasing_line_color=ACCENT,
                                     decreasing_line_color=MUTED, showlegend=False), 1, 1)
    else:
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close", line=dict(color=INK, width=2),
                                 hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=df["date"], y=df["deliv_pct"], name="Delivery %", marker_color="#9ec5f4",
                         hovertemplate="%{x|%d %b %Y}<br>delivery %{y:.1f}%<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["baseline_60"], name="60-day baseline",
                             line=dict(color=ORANGE, width=2),
                             hovertemplate="baseline %{y:.1f}%<extra></extra>"), 2, 1)
    if events is not None and len(events):
        for direction, sym_, lab in [("up", "triangle-up", "Up-move event"), ("down", "triangle-down", "Down-move event")]:
            e = events[events["direction"] == direction]
            if e.empty:
                continue
            fig.add_trace(go.Scatter(
                x=e["event_date"], y=e["event_close"], mode="markers", name=lab,
                marker=dict(symbol=sym_, size=16, color=[GROUP_COLORS.get(g, MUTED) for g in e["group"]],
                            line=dict(color=INK, width=1.5)),
                customdata=np.c_[e["group"], e["move_pct"], e["event_deliv"], e["event_rel_deliv"]],
                hovertemplate=("<b>%{x|%d %b %Y}</b><br>move %{customdata[1]:+.1f}%<br>"
                               "delivery %{customdata[2]:.1f}% (%{customdata[3]:+.1f} pp vs baseline)"
                               "<br>%{customdata[0]} conviction<extra></extra>")), 1, 1)
            for _, r in e.iterrows():
                fig.add_vrect(x0=r["start_date"], x1=r["event_date"], fillcolor=ACCENT, opacity=0.07,
                              line_width=0, row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_yaxes(ticksuffix="%", range=[0, 100], row=2, col=1)
    fig.update_annotations(font=dict(size=15, color=INK2), x=0, xanchor="left")
    return _style(fig, symbol, height=height)


def gauge(score: float | None, band_label: str | None) -> go.Figure:
    val = 0 if score is None or pd.isna(score) else float(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, number=dict(font=dict(size=56, color=INK), valueformat=".0f"),
        title=dict(text=f"Conviction Score<br><span style='font-size:20px;color:{INK2}'>{band_label or 'n/a'}</span>",
                   font=dict(size=20)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 30, 70, 100], tickfont=dict(color=MUTED)),
                   bar=dict(color=INK, thickness=0.25),
                   steps=[dict(range=[0, 30], color=BAND_COLORS["Low"]),
                          dict(range=[30, 70], color="#cde2fb"),
                          dict(range=[70, 100], color=BAND_COLORS["High"])])))
    return _style(fig, None, height=300)


def event_pair(paths: dict, title: str = "", reveal: bool = True, horizon: int = 60) -> go.Figure:
    """paths: label -> DataFrame(day, index) with day 0 = event date and index = 100 at day 0.
    If reveal is False, only the run-up (day <= 0) is drawn."""
    fig = go.Figure()
    for (label, p), color in zip(paths.items(), [ACCENT, ORANGE]):
        q = p if reveal else p[p["day"] <= 0]
        fig.add_trace(go.Scatter(x=q["day"], y=q["index"], name=label, mode="lines",
                                 line=dict(color=color, width=3),
                                 hovertemplate=f"{label}<br>day %{{x}}: %{{y:.1f}}<extra></extra>"))
        if reveal and len(q) and q["day"].max() > 0:
            last = q.iloc[-1]
            fig.add_annotation(x=last["day"], y=last["index"], text=f"{last['index'] - 100:+.0f}%",
                               showarrow=False, xanchor="left", xshift=8, font=dict(size=18, color=INK))
    fig.add_vline(x=0, line=dict(color=AXIS, dash="dot"))
    fig.add_hline(y=100, line=dict(color=AXIS, width=1))
    fig.update_xaxes(title="Trading days (0 = day the move completed)",
                     range=[None, horizon + 8] if reveal else [None, 3])
    fig.update_yaxes(title="Price, indexed to 100 on event day")
    return _style(fig, title, height=480)


def event_path(prices: pd.DataFrame, ev: pd.Series, post: int = 60, pre: int | None = None) -> pd.DataFrame:
    """Price path of one event indexed to 100 on the event date.
    prices: rows of one symbol with columns date, close (sorted). Day 0 = event date."""
    p = prices.sort_values("date").reset_index(drop=True)
    t = int(p.index[p["date"] == ev["event_date"]][0])
    m = int(p.index[p["date"] == ev["start_date"]][0])
    lo = m - (pre or 0)
    seg = p.iloc[max(lo, 0):t + post + 1]
    return pd.DataFrame({"day": seg.index - t, "index": seg["close"].to_numpy() / p.loc[t, "close"] * 100})
