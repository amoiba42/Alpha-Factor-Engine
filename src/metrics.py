"""Performance metrics. One methodology for every strategy.

Conventions
-----------
* Inputs are daily simple returns of a dollar-neutral, self-financing long/short book
  (return on gross capital). No risk-free rate is subtracted: short-sale proceeds are
  assumed to finance the long leg and rebates/financing costs are ignored (documented).
* Annualized return = mean daily return * 252 (arithmetic); CAGR is reported separately.
* Sharpe = mean / std * sqrt(252) of daily net returns.
* Information Ratio = mean(r - b) / std(r - b) * sqrt(252) against a named benchmark b.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

ANN = 252


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown of compounded wealth from its running peak (<= 0)."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1.0


def sharpe(returns: pd.Series, ann: int = ANN) -> float:
    r = returns.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else np.nan


def information_ratio(returns: pd.Series, benchmark: pd.Series, ann: int = ANN) -> float:
    active = (returns - benchmark.reindex(returns.index)).dropna()
    sd = active.std()
    return float(active.mean() / sd * np.sqrt(ann)) if sd > 0 else np.nan


def rolling_sharpe(returns: pd.Series, window: int = 126, ann: int = ANN) -> pd.Series:
    r = returns.rolling(window, min_periods=window)
    return r.mean() / r.std() * np.sqrt(ann)


def market_beta(returns: pd.Series, market: pd.Series, hac_lags: int = 5) -> dict[str, float]:
    """OLS beta of strategy returns on market returns with HAC standard errors."""
    df = pd.concat([returns.rename("r"), market.rename("m")], axis=1).dropna()
    if len(df) < 30:
        return {"beta": np.nan, "beta_t": np.nan, "alpha_ann": np.nan, "alpha_t": np.nan}
    res = sm.OLS(df["r"], sm.add_constant(df["m"])).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return {"beta": float(res.params["m"]), "beta_t": float(res.tvalues["m"]),
            "alpha_ann": float(res.params["const"] * ANN), "alpha_t": float(res.tvalues["const"])}


def performance_summary(daily: pd.DataFrame, benchmarks: dict[str, pd.Series],
                        market: pd.Series | None = None, ann: int = ANN) -> dict[str, float]:
    """Metrics for one strategy over the rows of `daily` (output of run_backtest)."""
    r = daily["net_return"].dropna()
    n = len(r)
    wealth = (1.0 + r).prod()
    out = {
        "n_days": n,
        "cumulative_return": wealth - 1.0,
        "cagr": wealth ** (ann / n) - 1.0 if n and wealth > 0 else np.nan,
        "ann_return": r.mean() * ann,
        "ann_volatility": r.std() * np.sqrt(ann),
        "sharpe": sharpe(r, ann),
        "sharpe_gross": sharpe(daily["gross_return"], ann),
        "max_drawdown": drawdown(r).min(),
        "win_rate": (r > 0).mean(),
        "avg_daily_return": r.mean(),
        "daily_volatility": r.std(),
        "avg_daily_turnover": daily["turnover"].mean(),
        "ann_turnover": daily["turnover"].mean() * ann,
        "ann_cost_drag": (daily["transaction_cost"] + daily["slippage"]).mean() * ann,
        "avg_gross_exposure": daily["gross_exposure"].mean(),
        "avg_net_exposure": daily["net_exposure"].mean(),
        "max_abs_net_exposure": daily["net_exposure"].abs().max(),
    }
    for name, b in benchmarks.items():
        out[f"ir_vs_{name}"] = information_ratio(r, b, ann)
    if market is not None:
        out.update({f"market_{k}": v for k, v in market_beta(r, market).items()})
    return out
