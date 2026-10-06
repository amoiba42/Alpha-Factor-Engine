"""Parameter selection and the Ridge signal ensemble.

Selection protocol (no test-period information is used):
* factor lookbacks are chosen by validation-period mean Rank IC against the
  execution-aligned forward return used by the strategy;
* the Ridge penalty is chosen by validation mean Rank IC of the model trained on the
  training period;
* the final model is refit on train + validation with the chosen penalty and only then
  applied to the test period.
Every label window is purged at period ends so no label uses prices beyond its period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .evaluation import period_dates, rank_ic
from .preprocessing import zscore_cs


def select_window(candidates: dict[int, pd.DataFrame], target_fwd: pd.DataFrame,
                  train: tuple[str, str], validation: tuple[str, str], purge: int) -> tuple[int, pd.DataFrame]:
    """Pick the lookback with the highest validation mean Rank IC; report train IC too."""
    rows = []
    for w, sig in candidates.items():
        ic = rank_ic(sig, target_fwd)
        tr = ic.reindex(period_dates(ic.index, *train, purge=purge))
        va = ic.reindex(period_dates(ic.index, *validation, purge=purge))
        rows.append({"window": w, "train_mean_ic": tr.mean(), "validation_mean_ic": va.mean(),
                     "validation_ic_ir": va.mean() / va.std()})
    tab = pd.DataFrame(rows).set_index("window")
    return int(tab["validation_mean_ic"].idxmax()), tab


def stack_features(signals: dict[str, pd.DataFrame], dates: pd.DatetimeIndex,
                   target: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray | None, pd.MultiIndex]:
    """Stack wide signals into a (n_obs, n_factors) design matrix over `dates`.

    Rows require every signal (and the target, if given) to be non-missing.
    """
    names = list(signals)
    cols = {n: signals[n].reindex(dates).stack(future_stack=True) for n in names}
    df = pd.DataFrame(cols)
    if target is not None:
        df["__y"] = target.reindex(dates).stack(future_stack=True)
    df = df.dropna()
    X = df[names].to_numpy()
    y = df["__y"].to_numpy() if target is not None else None
    return X, y, df.index


def fit_ridge(signals: dict[str, pd.DataFrame], target: pd.DataFrame, dates: pd.DatetimeIndex,
              alpha: float) -> Ridge:
    X, y, _ = stack_features(signals, dates, target)
    return Ridge(alpha=alpha, fit_intercept=True).fit(X, y)


def predict_composite(model: Ridge, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Composite score = sum_k coef_k * signal_k, re-standardized cross-sectionally.

    Only stocks with all signals present get a score. The intercept shifts every stock
    equally and is dropped (it cannot change cross-sectional rankings).
    """
    names = list(signals)
    valid = np.logical_and.reduce([signals[n].notna() for n in names])
    score = sum(model.coef_[i] * signals[n] for i, n in enumerate(names))
    return zscore_cs(score.where(valid))


def equal_weight_composite(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Baseline: average of the final signals, re-standardized cross-sectionally."""
    names = list(signals)
    valid = np.logical_and.reduce([signals[n].notna() for n in names])
    avg = sum(signals[n] for n in names) / len(names)
    return zscore_cs(avg.where(valid))


def select_ridge_alpha(signals: dict[str, pd.DataFrame], target: pd.DataFrame, target_fwd: pd.DataFrame,
                       train: tuple[str, str], validation: tuple[str, str], alphas: list[float],
                       purge: int) -> tuple[float, pd.DataFrame]:
    """Fit on purged train dates for each alpha; score by validation mean Rank IC.

    Ties (within 1e-6) go to the larger alpha, i.e. the more regularized model.
    """
    idx = target.index
    train_dates = period_dates(idx, *train, purge=purge)
    val_dates = period_dates(idx, *validation, purge=purge)
    rows = []
    for a in alphas:
        m = fit_ridge(signals, target, train_dates, a)
        comp = predict_composite(m, signals)
        ic = rank_ic(comp.reindex(val_dates), target_fwd.reindex(val_dates))
        rows.append({"alpha": a, "validation_mean_ic": ic.mean(), "validation_ic_ir": ic.mean() / ic.std(),
                     **{f"coef_{n}": c for n, c in zip(signals, m.coef_)}})
    tab = pd.DataFrame(rows).set_index("alpha")
    best = tab["validation_mean_ic"].max()
    chosen = max(a for a in alphas if tab.loc[a, "validation_mean_ic"] >= best - 1e-6)
    return float(chosen), tab
