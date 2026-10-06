"""Portfolio-engineering extensions evaluated on the frozen baseline signal (notebook 05).

This module does not change the baseline pipeline. It rebuilds the exact frozen Ridge
ensemble from the cached data and the selections recorded in results/tables, then
re-uses the same backtest engine with different portfolio-construction rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_config, table_relpath
from .data import build_panel
from .evaluation import period_dates
from .factors import FACTORS, build_factor
from .models import fit_ridge, predict_composite, select_ridge_alpha
from .portfolio import MIN_PER_LEG, long_short_weights
from .preprocessing import (build_universe, daily_returns, forward_returns, neutralize_cs, rank_gauss_cs,
                            rolling_beta, zscore_cs)

TABLES = PROJECT_ROOT / "results" / "tables"


def _read_table(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(TABLES / table_relpath(name), **kw)


def frozen_choices() -> dict:
    """Research decisions recorded by the frozen baseline run (validation-selected)."""
    sel = _read_table("lookback_selection").query("chosen")
    var = _read_table("design_variant_selection").query("chosen").iloc[0]
    return {"windows": {r.factor: int(r.window) for r in sel.itertuples()},
            "beta_neutral": bool(var["beta_neutral"]),
            "rebalance_every": int(var["rebalance_every"])}


def rebuild_baseline_inputs() -> dict:
    """Recreate returns, benchmark, and the frozen ensemble composite exactly as the baseline.

    Ridge: alpha selected on validation; train-period model scores train/validation dates,
    train+validation refit scores test dates (both purged), as in run_pipeline.py.
    """
    cfg = load_config()
    ch = frozen_choices()
    sp = cfg["splits"]
    train, val, test = tuple(sp["train"]), tuple(sp["validation"]), tuple(sp["test"])
    h, lag = cfg["evaluation"]["selection_horizon"], cfg["portfolio"]["execution_lag"]
    purge = h + lag

    panel = build_panel(cfg)
    adj = panel["wide"]["adj_close"]
    rets = daily_returns(adj)
    bench = panel["benchmark_returns"]
    universe = build_universe(panel["wide"], panel["membership"], cfg["universe"])
    dates = adj.index

    signals = {n: build_factor(n, ch["windows"][n], adj, rets, universe, cfg)["signal"] for n in FACTORS}
    if ch["beta_neutral"]:
        b = rolling_beta(rets, bench).where(universe)
        signals = {n: zscore_cs(neutralize_cs(s, b)) for n, s in signals.items()}

    target_fwd = forward_returns(adj, h, lag=lag).where(universe)
    target = rank_gauss_cs(target_fwd)
    alpha, _ = select_ridge_alpha(signals, target, target_fwd, train, val, cfg["model"]["ridge_alphas"], purge)
    m_train = fit_ridge(signals, target, period_dates(dates, *train, purge=purge), alpha)
    m_final = fit_ridge(signals, target, period_dates(dates, train[0], val[1], purge=purge), alpha)
    t0 = pd.Timestamp(test[0])
    c_tr, c_fi = predict_composite(m_train, signals), predict_composite(m_final, signals)
    composite = pd.concat([c_tr.loc[c_tr.index < t0], c_fi.loc[c_fi.index >= t0]])
    return {"cfg": cfg, "choices": ch, "alpha": alpha, "returns": rets, "benchmark": bench,
            "composite": composite, "periods": {"train": train, "validation": val, "test": test}}


def hysteresis_weights(signal: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, entry_q: float,
                       exit_q: float, gross: float = 1.0, min_per_leg: int = MIN_PER_LEG) -> pd.DataFrame:
    """Dollar-neutral long/short weights with a turnover buffer, on rebalance dates only.

    A stock enters the long (short) book when its cross-sectional percentile rank is in the
    top (bottom) `entry_q`; a held stock stays until it drops out of the top (bottom) `exit_q`
    (exit_q > entry_q). Stocks with no signal that day are dropped. Each leg is equal-weighted
    at gross/2, so long exposure = short exposure on every rebalance.

    Book membership is path dependent, so this loops over rebalance dates (about one
    thousand); every step is vectorized across stocks.
    """
    if not 0 < entry_q <= exit_q < 0.5:
        raise ValueError("need 0 < entry_q <= exit_q < 0.5")
    s = signal.reindex(rebalance_dates)
    pct = s.rank(axis=1, pct=True).to_numpy()
    valid = ~np.isnan(pct)
    pct = np.nan_to_num(pct, nan=0.5)
    long_ = np.zeros(pct.shape[1], dtype=bool)
    short = np.zeros(pct.shape[1], dtype=bool)
    W = np.zeros_like(pct)
    for i in range(len(rebalance_dates)):
        v = valid[i]
        p = pct[i]
        long_ = v & ((p > 1 - entry_q) | (long_ & (p > 1 - exit_q)))
        short = v & ((p <= entry_q) | (short & (p <= exit_q)))
        nl, ns = long_.sum(), short.sum()
        if nl >= min_per_leg and ns >= min_per_leg:
            W[i] = np.where(long_, gross / 2 / nl, 0.0) - np.where(short, gross / 2 / ns, 0.0)
        else:  # flat book; buffer state resets
            long_[:] = False
            short[:] = False
    return pd.DataFrame(W, index=rebalance_dates, columns=signal.columns)


def quantile_weights(signal: pd.DataFrame, quantile: float, gross: float = 1.0) -> pd.DataFrame:
    """Baseline equal-weighted top/bottom-quantile book (thin wrapper for clarity)."""
    return long_short_weights(signal, quantile, "equal", gross)
