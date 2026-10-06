"""Price-volume alpha factors.

Three representations are kept distinct throughout the project:

* raw factor          - economic quantity computed from prices through close t
                        (e.g. past 5-day return, 20-day volatility);
* standardized factor - raw factor winsorized and z-scored across the day's universe;
* final signal        - standardized factor times its prior direction, so that higher
                        always means higher expected return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import standardize

FACTORS = ("reversal", "momentum", "volatility")


def reversal_raw(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Past `window`-day return, close t-window -> close t. Reversal bets against it."""
    return adj_close / adj_close.shift(window) - 1.0


def momentum_raw(adj_close: pd.DataFrame, window: int, skip: int = 0) -> pd.DataFrame:
    """Return from close t-window to close t-skip (skips the most recent `skip` days)."""
    if skip >= window:
        raise ValueError("skip must be smaller than window")
    return adj_close.shift(skip) / adj_close.shift(window) - 1.0


def volatility_raw(returns: pd.DataFrame, window: int, annualization: int = 252) -> pd.DataFrame:
    """Annualized std of daily returns over t-window+1..t (requires 80% valid days)."""
    vol = returns.rolling(window, min_periods=int(np.ceil(0.8 * window))).std()
    return vol * np.sqrt(annualization)


def raw_factor(name: str, window: int, adj_close: pd.DataFrame, returns: pd.DataFrame,
               fcfg: dict) -> pd.DataFrame:
    if name == "reversal":
        return reversal_raw(adj_close, window)
    if name == "momentum":
        return momentum_raw(adj_close, window, fcfg["momentum_skip"])
    if name == "volatility":
        return volatility_raw(returns, window)
    raise KeyError(name)


def candidate_windows(fcfg: dict) -> dict[str, list[int]]:
    return {"reversal": fcfg["reversal_windows"], "momentum": fcfg["momentum_windows"],
            "volatility": fcfg["volatility_windows"]}


def build_factor(name: str, window: int, adj_close: pd.DataFrame, returns: pd.DataFrame,
                 universe: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Return {'raw', 'standardized', 'signal'} frames for one factor/window."""
    raw = raw_factor(name, window, adj_close, returns, cfg["factors"])
    std = standardize(raw, universe, cfg["preprocessing"])
    signal = std * cfg["factors"]["directions"][name]
    return {"raw": raw.where(universe), "standardized": std, "signal": signal}
