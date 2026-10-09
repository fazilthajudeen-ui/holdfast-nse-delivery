"""Event-driven portfolio backtest (spec section 11).

  * entry at the T+1 open (G1-consistent), exit at the triple-barrier exit (barrier fill, gap open,
    or close at the time barrier) - prices are split-adjusted and consistent with the labels;
  * equal weight: each new position gets 1/max_positions of the previous close's equity, at most 20
    concurrent positions; when more signals than free slots arrive on one day, higher `priority`
    wins (ties broken by a seeded random draw);
  * a slot freed on day D is reusable from D+1 (entries are processed before exits each day);
  * costs from config/costs.yaml on both legs plus slippage by liquidity bucket;
  * daily mark-to-market at closes; cash earns nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from impi.backtest import costs as C
from impi.common import config


def run_backtest(signals: pd.DataFrame, closes: dict[str, pd.Series], calendar: np.ndarray,
                 seed: int, max_positions: int | None = None, capital: float | None = None,
                 cash_yield: float | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Idle cash earns `cash_yield` (default: the risk-free rate in costs.yaml, a liquid-fund
    assumption - D-022), so a strategy that is mostly in cash is not penalised in its Sharpe ratio."""
    cc = config()["costs"]
    max_pos = int(max_positions or cc["max_positions"])
    cash = float(capital or cc["initial_capital"])
    cy = float(cc["risk_free_annual"] if cash_yield is None else cash_yield)
    cash_daily = (1 + cy) ** (1 / 252) - 1
    if signals.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    s = signals.dropna(subset=["entry_date", "exit_date", "entry_px", "exit_px"]).copy()
    s = s[(s["entry_px"] > 0) & (s["exit_px"] > 0)]
    rng = np.random.default_rng(seed)
    s["_tie"] = rng.random(len(s))
    if "priority" not in s:
        s["priority"] = 0.0
    s = s.sort_values(["entry_date", "priority", "_tie"], ascending=[True, False, False])
    by_entry = {d: g for d, g in s.groupby("entry_date")}
    cal = pd.DatetimeIndex(calendar)
    start, end = s["entry_date"].min(), s["exit_date"].max()
    days = cal[(cal >= start) & (cal <= end)]
    open_pos: list[dict] = []
    trades, equity_curve = [], {}
    last_close: dict[str, float] = {}
    prev_equity = cash
    for d in days:
        if cash > 0:
            cash *= 1 + cash_daily
        # entries at today's open
        for _, sig in by_entry.get(d, pd.DataFrame()).iterrows():
            if len(open_pos) >= max_pos:
                break
            if any(p["symbol"] == sig["symbol"] for p in open_pos):
                continue                                   # one position per stock at a time
            alloc = prev_equity / max_pos
            notional = min(alloc, cash / (1 + 0.0025))     # keep room for buy costs
            if notional <= 1000:
                continue
            slip = C.slippage_bps(sig.get("liq_bucket")) / 1e4
            fill = sig["entry_px"] * (1 + slip)
            bc = C.buy_cost(notional)
            cash -= notional + bc
            open_pos.append({"symbol": sig["symbol"], "shares": notional / fill, "cost_basis": notional + bc,
                             "entry_date": d, "exit_date": sig["exit_date"], "exit_px": sig["exit_px"],
                             "slip": slip, "bucket": sig.get("liq_bucket"), "family": sig.get("family"),
                             "event_date": sig.get("date")})
        # exits
        still = []
        for p in open_pos:
            if p["exit_date"] <= d:
                fill = p["exit_px"] * (1 - p["slip"])
                proceeds = p["shares"] * fill
                sc = C.sell_cost(proceeds)
                cash += proceeds - sc
                pnl = proceeds - sc - p["cost_basis"]
                trades.append({"symbol": p["symbol"], "event_date": p["event_date"], "entry_date": p["entry_date"],
                               "exit_date": d, "notional": p["cost_basis"], "pnl": pnl,
                               "net_ret": pnl / p["cost_basis"], "bucket": p["bucket"],
                               "hold_days": int(((cal >= p["entry_date"]) & (cal <= d)).sum())})
            else:
                still.append(p)
        open_pos = still
        # mark to market
        mtm = 0.0
        for p in open_pos:
            px = closes.get(p["symbol"])
            v = px.get(d, np.nan) if px is not None else np.nan
            if np.isfinite(v):
                last_close[p["symbol"]] = v
            mtm += p["shares"] * last_close.get(p["symbol"], p["cost_basis"] / p["shares"])
        equity = cash + mtm
        equity_curve[d] = equity
        prev_equity = equity
    eq = pd.Series(equity_curve)
    ret = eq.pct_change()
    ret.iloc[0] = eq.iloc[0] / float(capital or cc["initial_capital"]) - 1
    tr = pd.DataFrame(trades)
    if not tr.empty:
        traded = tr["notional"].sum() + (tr["notional"] + tr["pnl"]).sum()
        ret.attrs["turnover_per_year"] = float(traded / eq.mean() / (len(eq) / 252))
    return ret, tr
