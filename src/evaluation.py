"""Factor evaluation: Rank IC, IC statistics, decay and factor correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

MIN_NAMES = 20  # minimum stocks with valid signal and return for a daily IC


# --------------------------------------------------------------------------------------
# Period handling
# --------------------------------------------------------------------------------------
def period_dates(index: pd.DatetimeIndex, start: str, end: str, purge: int = 0) -> pd.DatetimeIndex:
    """Dates in [start, end], dropping the last `purge` trading days of the period.

    A label at t that depends on prices up to t+purge must not use prices after `end`.
    Dropping the final `purge` dates keeps every label inside the period.
    """
    sel = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    return sel[:-purge] if purge > 0 else sel


# --------------------------------------------------------------------------------------
# Information coefficient
# --------------------------------------------------------------------------------------
def rowwise_corr(a: pd.DataFrame, b: pd.DataFrame, min_names: int = MIN_NAMES) -> pd.Series:
    """Pearson correlation between rows of `a` and `b` over jointly valid columns."""
    valid = a.notna() & b.notna()
    x, y = a.where(valid), b.where(valid)
    x = x.sub(x.mean(axis=1), axis=0)
    y = y.sub(y.mean(axis=1), axis=0)
    num = (x * y).sum(axis=1)
    den = np.sqrt((x ** 2).sum(axis=1) * (y ** 2).sum(axis=1))
    corr = num / den.replace(0.0, np.nan)
    return corr.where(valid.sum(axis=1) >= min_names)


def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_names: int = MIN_NAMES) -> pd.Series:
    """Daily Spearman rank IC: Pearson correlation of cross-sectional ranks, ranking only
    the stocks where both signal and forward return are available."""
    valid = signal.notna() & fwd.notna()
    rs = signal.where(valid).rank(axis=1)
    rf = fwd.where(valid).rank(axis=1)
    return rowwise_corr(rs, rf, min_names)


def newey_west_tstat(x: pd.Series, lags: int) -> float:
    """t-statistic of the mean with Newey-West (HAC) standard errors.

    Overlapping h-day forward returns make daily ICs autocorrelated; HAC with lags >= h-1
    corrects the naive t-stat.
    """
    x = x.dropna()
    if len(x) < 10:
        return np.nan
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)})
    return float(res.tvalues[0])


def ic_summary(ic: pd.Series, horizon: int) -> dict[str, float]:
    ic = ic.dropna()
    sd = ic.std()
    return {
        "mean_ic": ic.mean(),
        "median_ic": ic.median(),
        "ic_std": sd,
        "ic_ir": ic.mean() / sd if sd > 0 else np.nan,
        "pct_positive": (ic > 0).mean(),
        "t_stat_nw": newey_west_tstat(ic, horizon),
        "n_days": len(ic),
    }


def ic_table(signals: dict[str, pd.DataFrame], fwd: dict[int, pd.DataFrame],
             dates: pd.DatetimeIndex | None = None,
             period_end_purge: tuple[str, str] | None = None) -> pd.DataFrame:
    """IC summary for every (factor, horizon).

    If `period_end_purge=(start, end)` is given, each horizon h uses only dates whose
    h-day forward return ends inside [start, end].
    """
    rows = []
    for name, sig in signals.items():
        for h, f in fwd.items():
            ic = rank_ic(sig, f)
            if period_end_purge is not None:
                ic = ic.reindex(period_dates(ic.index, *period_end_purge, purge=h))
            elif dates is not None:
                ic = ic.reindex(dates)
            rows.append({"factor": name, "horizon": h, **ic_summary(ic, h)})
    return pd.DataFrame(rows)


def decay_table(ic_tab: pd.DataFrame, stat: str = "mean_ic") -> pd.DataFrame:
    """Factor x horizon pivot of an IC statistic."""
    out = ic_tab.pivot(index="factor", columns="horizon", values=stat)
    out.columns = [f"{h}D" for h in out.columns]
    return out


def ic_by_year(ic: pd.Series) -> pd.DataFrame:
    g = ic.dropna().groupby(ic.dropna().index.year)
    return pd.DataFrame({"mean_ic": g.mean(), "ic_std": g.std(), "pct_positive": g.apply(lambda s: (s > 0).mean()),
                         "n_days": g.size()})


def ic_concentration(ic: pd.Series, top_frac: float = 0.05) -> dict[str, float]:
    """How much the mean IC depends on extreme days: mean IC after dropping the top and
    bottom `top_frac` of days, and the share of the total IC sum from the top days."""
    ic = ic.dropna().sort_values()
    k = int(len(ic) * top_frac)
    trimmed = ic.iloc[k:len(ic) - k] if k else ic
    return {"trimmed_mean_ic": trimmed.mean(),
            "share_of_ic_sum_from_top_days": ic.iloc[-k:].sum() / ic.sum() if k and ic.sum() != 0 else np.nan}


# --------------------------------------------------------------------------------------
# Factor relationships
# --------------------------------------------------------------------------------------
def mean_cs_correlation(frames: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
                        method: str = "spearman") -> pd.DataFrame:
    """Time-average of daily cross-sectional correlations between factors."""
    names = list(frames)
    prep = {k: v.reindex(dates) for k, v in frames.items()}
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = prep[a], prep[b]
            if method == "spearman":
                valid = x.notna() & y.notna()
                x, y = x.where(valid).rank(axis=1), y.where(valid).rank(axis=1)
            c = rowwise_corr(x, y).mean()
            out.loc[a, b] = out.loc[b, a] = c
    return out
