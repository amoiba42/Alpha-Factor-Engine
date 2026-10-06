"""Event-driven daily backtest with execution lag, weight drift, turnover and costs.

Timing (all at daily closes):
* signal and target weights are formed from data through close s;
* on rebalance signal dates s, the trade executes at close s + `execution_lag`;
* holdings after a trade at close t earn the close t -> close t+1 return;
* between rebalances positions are held (share counts fixed), so weights drift.

Costs are charged per dollar traded: `turnover_t = sum_i |w_target_i - w_drifted_i|`,
transaction cost = turnover * (commission + spread) bps, slippage = turnover * slippage bps.
Net return_t = gross return_t - transaction cost_t - slippage_t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_schedule(index: pd.DatetimeIndex, start: pd.Timestamp | None, every: int) -> pd.DatetimeIndex:
    """Signal dates on which the portfolio is rebalanced: every `every`-th trading day
    from `start` (inclusive)."""
    idx = index if start is None else index[index >= start]
    return idx[::every]


def run_backtest(target_weights: pd.DataFrame, returns: pd.DataFrame, rebalance_dates: pd.DatetimeIndex,
                 execution_lag: int = 1, cost_bps: float = 3.0, slippage_bps: float = 3.0,
                 end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the strategy.

    Parameters
    ----------
    target_weights : date x ticker weights indexed by *signal* date.
    returns        : date x ticker simple daily returns (close t-1 -> close t).
    rebalance_dates: signal dates on which to trade.

    Returns
    -------
    (daily, holdings): `daily` has gross/net returns, costs, turnover, exposures and leg
    contributions per date; `holdings` are end-of-day weights after trading.
    A held stock with a missing return that day contributes 0 (position kept as is).
    """
    idx = returns.index
    if end is not None:
        idx = idx[idx <= end]
    tickers = returns.columns
    R = returns.reindex(index=idx, columns=tickers).to_numpy()
    W = target_weights.reindex(columns=tickers).fillna(0.0)

    sig_pos = idx.get_indexer(rebalance_dates)
    found = sig_pos >= 0
    sig_dates = rebalance_dates[found]
    exec_pos = sig_pos[found] + execution_lag
    keep = exec_pos < len(idx)
    targets = W.reindex(sig_dates[keep]).fillna(0.0).to_numpy()
    trade_at = dict(zip(exec_pos[keep].tolist(), targets))

    T, N = R.shape
    h = np.zeros(N)
    out = np.full((T, 10), np.nan)
    H = np.zeros((T, N))
    first_trade = min(trade_at) if trade_at else T
    for t in range(T):
        r = np.nan_to_num(R[t], nan=0.0)
        contrib = h * r
        gross_ret = contrib.sum()
        long_c = contrib[h > 0].sum()
        short_c = contrib[h < 0].sum()
        # drift: dollar holdings grow with their own return, capital with the portfolio return
        denom = 1.0 + gross_ret
        h = h * (1.0 + r) / denom if denom > 0 else h * (1.0 + r)
        turnover = 0.0
        if t in trade_at:
            w = trade_at[t]
            turnover = np.abs(w - h).sum()
            h = w.copy()
        tc = turnover * cost_bps * 1e-4
        sl = turnover * slippage_bps * 1e-4
        out[t] = [gross_ret, tc, sl, gross_ret - tc - sl, turnover, long_c, short_c,
                  h[h > 0].sum(), h[h < 0].sum(), (h != 0).sum()]
        H[t] = h

    daily = pd.DataFrame(out, index=idx, columns=[
        "gross_return", "transaction_cost", "slippage", "net_return", "turnover",
        "long_contribution", "short_contribution", "long_exposure", "short_exposure", "n_positions"])
    daily["gross_exposure"] = daily["long_exposure"] - daily["short_exposure"]
    daily["net_exposure"] = daily["long_exposure"] + daily["short_exposure"]
    # nothing is held before the first trade: drop the leading empty period
    daily = daily.iloc[first_trade:]
    holdings = pd.DataFrame(H, index=idx, columns=tickers).iloc[first_trade:]
    return daily, holdings
