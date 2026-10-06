import numpy as np
import pandas as pd

from src.data import clean_prices, membership_mask
from src.models import (equal_weight_composite, fit_ridge, predict_composite, select_ridge_alpha,
                        stack_features)
from src.preprocessing import forward_returns, rank_gauss_cs


def _signals(prices, seed=9):
    rng = np.random.default_rng(seed)
    fwd = forward_returns(prices, 5)
    noise = lambda: pd.DataFrame(rng.normal(size=prices.shape), index=prices.index, columns=prices.columns)  # noqa: E731
    target = rank_gauss_cs(fwd)
    # "a" is informative about the target, "b" is pure noise
    return {"a": target.fillna(0) + 2 * noise(), "b": noise()}, target, fwd


def test_ridge_learns_informative_signal(prices):
    sig, target, _ = _signals(prices)
    m = fit_ridge(sig, target, prices.index[:200], alpha=1.0)
    assert m.coef_[0] > 5 * abs(m.coef_[1])


def test_ridge_shrinks_with_alpha(prices):
    sig, target, _ = _signals(prices)
    small = fit_ridge(sig, target, prices.index[:200], alpha=1e-3)
    big = fit_ridge(sig, target, prices.index[:200], alpha=1e6)
    assert np.abs(big.coef_).sum() < np.abs(small.coef_).sum()


def test_fit_uses_only_given_dates(prices):
    sig, target, _ = _signals(prices)
    dates = prices.index[:150]
    m1 = fit_ridge(sig, target, dates, 1.0)
    t2 = target.copy()
    t2.iloc[150:] = -t2.iloc[150:]  # corrupt labels outside the training dates
    m2 = fit_ridge(sig, t2, dates, 1.0)
    assert np.allclose(m1.coef_, m2.coef_)


def test_stack_features_drops_incomplete_rows(prices):
    sig, target, _ = _signals(prices)
    sig["a"].iloc[0, 0] = np.nan
    X, y, idx = stack_features(sig, prices.index[:1], target)
    assert X.shape == (39, 2) and len(y) == 39


def test_alpha_selection_does_not_see_test_period(prices):
    sig, target, fwd = _signals(prices)
    train, val = (str(prices.index[0].date()), str(prices.index[149].date())), \
        (str(prices.index[150].date()), str(prices.index[229].date()))
    a1, t1 = select_ridge_alpha(sig, target, fwd, train, val, [0.1, 10.0], purge=5)
    sig2 = {k: v.copy() for k, v in sig.items()}
    for v in sig2.values():
        v.iloc[230:] = 99.0  # test period changed
    t3, f3 = target.copy(), fwd.copy()
    t3.iloc[230:], f3.iloc[230:] = 0.0, 0.0
    a2, t2 = select_ridge_alpha(sig2, t3, f3, train, val, [0.1, 10.0], purge=5)
    assert a1 == a2
    pd.testing.assert_frame_equal(t1, t2)


def test_composites_are_standardized(prices):
    sig, target, _ = _signals(prices)
    m = fit_ridge(sig, target, prices.index[:200], 1.0)
    for comp in (predict_composite(m, sig), equal_weight_composite(sig)):
        assert np.allclose(comp.mean(axis=1), 0, atol=1e-10)
        assert np.allclose(comp.std(axis=1, ddof=0), 1, atol=1e-10)


def test_membership_mask_point_in_time():
    snaps = pd.Series([["A", "B"], ["B", "C"]], index=pd.to_datetime(["2020-01-01", "2020-01-08"]))
    dates = pd.bdate_range("2020-01-01", "2020-01-10")
    m = membership_mask(snaps, dates, ["A", "B", "C"])
    assert m.loc["2020-01-07", "A"] and not m.loc["2020-01-07", "C"]
    assert not m.loc["2020-01-08", "A"] and m.loc["2020-01-08", "C"]
    assert m["B"].all()


def test_clean_prices_rules():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03", "2020-01-06"]),
        "ticker": ["A"] * 5,
        "open": [10, 10, 10, 10, 30], "high": [10, 10, 10, 10, 30], "low": [10, 10, 10, 10, 30],
        "close": [10, 10, 10, -1, 30], "adj_close": [10, 10, 11, 10, 30], "volume": [1, 1, 1, 1, 1],
    }).astype({c: float for c in ["open", "high", "low", "close", "adj_close", "volume"]})
    cfg = {"cleaning": {"max_abs_daily_return": 0.75}}
    out, actions = clean_prices(df, cfg)
    assert actions["duplicates_dropped"] == 1
    assert actions["non_positive_prices_set_nan"] == 1
    assert actions["extreme_return_days_set_nan"] == 1  # 10 -> 30 is +200%
    assert out["date"].is_unique
