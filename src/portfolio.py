"""Portfolio construction: dollar-neutral long/short weights and exposure diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PER_LEG = 5


def long_short_weights(signal: pd.DataFrame, quantile: float = 0.2, weighting: str = "equal",
                       gross: float = 1.0, min_per_leg: int = MIN_PER_LEG) -> pd.DataFrame:
    """Target weights from a cross-sectional signal (higher = more attractive).

    Long the top `quantile` of names, short the bottom `quantile`. Each leg carries
    gross/2 of capital, so long exposure = short exposure (dollar neutral) whenever both
    legs have at least `min_per_leg` names; otherwise all weights are zero that day.

    weighting='equal'  : equal weight within each leg.
    weighting='signal' : weight proportional to |signal - cross-sectional median| in each leg.
    """
    pct = signal.rank(axis=1, pct=True)
    long_m = pct > 1.0 - quantile
    short_m = pct <= quantile
    if weighting == "equal":
        lw = long_m.astype(float)
        sw = short_m.astype(float)
    elif weighting == "signal":
        dev = signal.sub(signal.median(axis=1), axis=0).abs()
        lw = dev.where(long_m, 0.0)
        sw = dev.where(short_m, 0.0)
    else:
        raise ValueError(weighting)
    lw = lw.div(lw.sum(axis=1).replace(0.0, np.nan), axis=0) * gross / 2
    sw = sw.div(sw.sum(axis=1).replace(0.0, np.nan), axis=0) * gross / 2
    ok = (long_m.sum(axis=1) >= min_per_leg) & (short_m.sum(axis=1) >= min_per_leg)
    w = (lw.fillna(0.0) - sw.fillna(0.0)).where(ok, 0.0, axis=0)
    return w.fillna(0.0)


def exposures(weights: pd.DataFrame) -> pd.DataFrame:
    """Daily long, short, gross and net exposure and position counts."""
    long_ = weights.clip(lower=0).sum(axis=1)
    short = weights.clip(upper=0).sum(axis=1)
    return pd.DataFrame({
        "long_exposure": long_,
        "short_exposure": short,
        "gross_exposure": long_ - short,
        "net_exposure": long_ + short,
        "n_long": (weights > 0).sum(axis=1),
        "n_short": (weights < 0).sum(axis=1),
    })


def ex_ante_beta(weights: pd.DataFrame, betas: pd.DataFrame) -> pd.Series:
    """Portfolio beta implied by holdings and trailing stock betas (estimated through t)."""
    b = betas.reindex_like(weights)
    held = weights != 0
    return (weights * b.where(held, 0.0)).sum(axis=1).where(held.any(axis=1))
