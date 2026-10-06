"""Returns, forward returns, universe construction and cross-sectional transforms.

All frames are wide (date x ticker). Cross-sectional operations act on each row (date)
independently, so they use only same-date information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------------------
def daily_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Simple return from close t-1 to close t, stored at t. NaN if either price missing."""
    return adj_close.pct_change(fill_method=None)


def forward_returns(adj_close: pd.DataFrame, horizon: int, lag: int = 0) -> pd.DataFrame:
    """Return earned from close (t+lag) to close (t+lag+horizon), stored at t.

    With lag=0 and horizon=1 this is exactly `return(t+1)`: the next-day return that a
    feature observed at close t is meant to predict.
    """
    if horizon < 1 or lag < 0:
        raise ValueError("horizon must be >= 1 and lag >= 0")
    start = adj_close.shift(-lag)
    end = adj_close.shift(-(lag + horizon))
    return end / start - 1.0


# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------
def build_universe(wide: dict[str, pd.DataFrame], membership: pd.DataFrame | None,
                   ucfg: dict) -> pd.DataFrame:
    """Boolean (date x ticker) tradable-universe mask using information through t only.

    Conditions on date t:
    * valid adj_close and positive volume on t,
    * point-in-time index membership on t (optional),
    * split-adjusted close >= min_price, if `min_price` is configured,
    * trailing `adv_window`-day median dollar volume (close * volume, through t) >= min_adv_usd,
    * at least `min_history` valid daily returns up to t.
    """
    close, volume, adj = wide["close"], wide["volume"], wide["adj_close"]
    dollar_vol = close * volume
    adv = dollar_vol.rolling(ucfg["adv_window"], min_periods=ucfg["adv_window"] // 2).median()
    n_hist = daily_returns(adj).notna().cumsum()
    mask = (
        adj.notna()
        & (volume > 0)
        & (adv >= ucfg["min_adv_usd"])
        & (n_hist >= ucfg["min_history"])
    )
    if ucfg.get("min_price") is not None:
        mask &= close >= ucfg["min_price"]
    if membership is not None and ucfg.get("point_in_time_membership", True):
        mask &= membership.reindex_like(mask).fillna(False).astype(bool)
    return mask


# --------------------------------------------------------------------------------------
# Cross-sectional transforms
# --------------------------------------------------------------------------------------
def winsorize_cs(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each row at its own lower/upper quantiles (NaNs ignored, stay NaN)."""
    q = df.quantile([lower, upper], axis=1)
    return df.clip(lower=q.loc[lower], upper=q.loc[upper], axis=0)


def zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (x - mean) / std. Rows with fewer than 2 values or zero std become NaN."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def rank_gauss_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise rank mapped to standard-normal quantiles (robust, unit-scale)."""
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1)
    u = r.sub(0.5).div(n, axis=0)
    return pd.DataFrame(stats.norm.ppf(u), index=df.index, columns=df.columns)


def standardize(raw: pd.DataFrame, universe: pd.DataFrame, pcfg: dict) -> pd.DataFrame:
    """Raw factor -> standardized factor, using only stocks in the universe on each date.

    Steps: restrict to universe, winsorize, then z-score (or rank-gaussianize), clip.
    """
    x = raw.where(universe)
    if pcfg.get("transform", "zscore") == "rank":
        z = rank_gauss_cs(x)
    else:
        z = zscore_cs(winsorize_cs(x, pcfg["winsor_lower"], pcfg["winsor_upper"]))
    clip = pcfg.get("z_clip")
    return z.clip(-clip, clip) if clip else z


def neutralize_cs(signal: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """Remove each date's cross-sectional linear exposure to `exposure` (e.g. market beta).

    Per row: residual of OLS(signal ~ 1 + exposure), computed in closed form.
    """
    valid = signal.notna() & exposure.notna()
    s, e = signal.where(valid), exposure.where(valid)
    s_dm = s.sub(s.mean(axis=1), axis=0)
    e_dm = e.sub(e.mean(axis=1), axis=0)
    slope = (s_dm * e_dm).sum(axis=1) / (e_dm ** 2).sum(axis=1).replace(0.0, np.nan)
    return s_dm.sub(e_dm.mul(slope, axis=0))


def rolling_beta(returns: pd.DataFrame, market: pd.Series, window: int = 252,
                 min_periods: int = 126) -> pd.DataFrame:
    """Trailing OLS beta of each stock on the market, using returns through t only."""
    m = market.reindex(returns.index)
    valid = returns.notna() & m.notna().to_numpy()[:, None]
    r = returns.where(valid)
    mm = pd.DataFrame(np.where(valid, m.to_numpy()[:, None], np.nan), index=r.index, columns=r.columns)
    roll = lambda x: x.rolling(window, min_periods=min_periods).mean()  # noqa: E731
    cov = roll(r * mm) - roll(r) * roll(mm)
    var = roll(mm * mm) - roll(mm) ** 2
    return cov / var
