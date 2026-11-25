"""Strategy metrics, deflated Sharpe ratio and White's Reality Check / Hansen SPA (spec section 11)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER = 0.5772156649015329


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min()) if len(equity) else np.nan


def portfolio_metrics(daily_ret: pd.Series, rf_annual: float = 0.0) -> dict:
    r = daily_ret.dropna()
    if len(r) < 2:
        return {}
    rf = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    ex = r - rf
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    sd = ex.std(ddof=1)
    down = ex[ex < 0]
    dd = np.sqrt((down ** 2).sum() / len(ex)) if len(ex) else np.nan
    mdd = max_drawdown(eq)
    return {"cagr": float(cagr), "ann_vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "sharpe": float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "sortino": float(ex.mean() / dd * np.sqrt(TRADING_DAYS)) if dd and dd > 0 else np.nan,
            "max_dd": mdd, "calmar": float(cagr / abs(mdd)) if mdd and mdd < 0 and np.isfinite(cagr) else np.nan,
            "total_return": float(eq.iloc[-1] - 1), "days": int(len(r)),
            "sharpe_daily": float(ex.mean() / sd) if sd > 0 else np.nan,
            "skew": float(stats.skew(r)), "kurtosis": float(stats.kurtosis(r, fisher=False))}


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    r = trades["net_ret"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_win = (trades.loc[r > 0, "pnl"]).sum()
    gross_loss = -(trades.loc[r <= 0, "pnl"]).sum()
    return {"n_trades": int(len(r)), "hit_rate": float((r > 0).mean()),
            "avg_win": float(wins.mean()) if len(wins) else np.nan,
            "avg_loss": float(losses.mean()) if len(losses) else np.nan,
            "win_loss_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.nan,
            "avg_net_ret": float(r.mean()), "avg_hold_days": float(trades["hold_days"].mean())}


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, sr_trials_var: float, n_trials: int) -> dict:
    """Bailey & Lopez de Prado (2014). sr, sr_trials_var in PER-PERIOD (daily) units; kurt is
    non-excess kurtosis. Returns the expected max SR under the null (SR0) and DSR = P(SR* > SR0)."""
    if n_trials < 2 or not np.isfinite(sr) or sr_trials_var <= 0 or n_obs < 3:
        return {"sr0": np.nan, "dsr": np.nan}
    sr0 = np.sqrt(sr_trials_var) * ((1 - EULER) * stats.norm.ppf(1 - 1 / n_trials)
                                     + EULER * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    return {"sr0": float(sr0), "dsr": float(stats.norm.cdf(z)), "z": float(z)}


def spa_tests(benchmark_ret: pd.Series, variant_rets: pd.DataFrame, reps: int, block: int, seed: int) -> dict:
    """Hansen SPA and White's Reality Check: does ANY variant beat the benchmark (higher mean daily
    return)? Losses are negative returns. Uses the `arch` implementation (stationary bootstrap)."""
    from arch.bootstrap import SPA, RealityCheck
    df = pd.concat([benchmark_ret.rename("bench"), variant_rets], axis=1).fillna(0.0)
    if df.shape[0] < 50 or df.shape[1] < 2:
        return {"spa_p_consistent": np.nan, "white_rc_p": np.nan}
    bench_loss = -df["bench"].to_numpy()
    model_loss = -df.drop(columns="bench").to_numpy()
    spa = SPA(bench_loss, model_loss, reps=reps, block_size=block, bootstrap="stationary", seed=seed)
    spa.compute()
    rc = RealityCheck(bench_loss, model_loss, reps=reps, block_size=block, bootstrap="stationary", seed=seed)
    rc.compute()
    return {"spa_p_consistent": float(spa.pvalues["consistent"]), "spa_p_lower": float(spa.pvalues["lower"]),
            "spa_p_upper": float(spa.pvalues["upper"]), "white_rc_p": float(rc.pvalues["upper"]),
            "n_variants": int(model_loss.shape[1]), "n_days": int(df.shape[0])}
