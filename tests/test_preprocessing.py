import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (build_universe, daily_returns, forward_returns, neutralize_cs,
                               rank_gauss_cs, rolling_beta, standardize, winsorize_cs, zscore_cs)


def test_daily_return_alignment(prices):
    r = daily_returns(prices)
    t = prices.index[10]
    expected = prices.loc[t, "T00"] / prices.iloc[9]["T00"] - 1
    assert r.loc[t, "T00"] == pytest.approx(expected)
    assert r.iloc[0].isna().all()


def test_forward_return_is_next_day_return(prices):
    r = daily_returns(prices)
    f1 = forward_returns(prices, 1)
    # forward_return(t) = return(t+1)
    pd.testing.assert_frame_equal(f1.iloc[:-1], r.shift(-1).iloc[:-1])
    assert f1.iloc[-1].isna().all()


def test_forward_return_horizon_and_lag(prices):
    f = forward_returns(prices, 5, lag=1)
    i = 20
    expected = prices.iloc[i + 6]["T03"] / prices.iloc[i + 1]["T03"] - 1
    assert f.iloc[i]["T03"] == pytest.approx(expected)
    assert f.iloc[-6:].isna().all().all()


def test_forward_return_does_not_change_past_when_future_changes(prices):
    f_before = forward_returns(prices, 5)
    p2 = prices.copy()
    p2.iloc[200:] *= 3.0  # shock after date 199
    f_after = forward_returns(p2, 5)
    # labels whose window ends before the shock are unchanged
    pd.testing.assert_frame_equal(f_before.iloc[:194], f_after.iloc[:194])


def test_winsorize_clips_each_row():
    df = pd.DataFrame([np.arange(101, dtype=float), np.arange(101, dtype=float) * 2])
    w = winsorize_cs(df, 0.05, 0.95)
    assert w.iloc[0].min() == pytest.approx(5.0)
    assert w.iloc[0].max() == pytest.approx(95.0)
    assert w.iloc[1].max() == pytest.approx(190.0)


def test_zscore_rows_mean0_std1(prices):
    z = zscore_cs(prices)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-10)
    assert np.allclose(z.std(axis=1, ddof=0), 1, atol=1e-10)


def test_zscore_ignores_nan_and_keeps_it():
    df = pd.DataFrame([[1.0, 2.0, np.nan, 3.0]])
    z = zscore_cs(df)
    assert np.isnan(z.iloc[0, 2])
    assert z.iloc[0, [0, 1, 3]].mean() == pytest.approx(0.0)


def test_rank_gauss_monotone_and_centered(prices):
    g = rank_gauss_cs(prices)
    row = prices.iloc[50]
    assert (g.iloc[50][row.sort_values().index].diff().dropna() > 0).all()
    assert g.mean(axis=1).abs().max() < 1e-10


def test_standardize_uses_universe_only(prices):
    uni = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    uni.iloc[:, :5] = False
    pcfg = {"winsor_lower": 0.01, "winsor_upper": 0.99, "transform": "zscore", "z_clip": 3.0}
    s = standardize(prices, uni, pcfg)
    assert s.iloc[:, :5].isna().all().all()
    # changing excluded stocks must not change standardized values of included stocks
    p2 = prices.copy()
    p2.iloc[:, :5] *= 1000
    pd.testing.assert_frame_equal(s, standardize(p2, uni, pcfg))


def test_neutralize_removes_exposure(prices):
    rng = np.random.default_rng(1)
    expo = pd.DataFrame(rng.normal(size=prices.shape), index=prices.index, columns=prices.columns)
    sig = 2 * expo + pd.DataFrame(rng.normal(size=prices.shape), index=prices.index, columns=prices.columns)
    res = neutralize_cs(sig, expo)
    corr = res.T.corrwith(expo.T)
    assert corr.abs().max() < 1e-8


def test_rolling_beta_recovers_true_beta():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=400)
    m = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    r = pd.DataFrame({"A": 1.5 * m + rng.normal(0, 0.001, 400), "B": -0.5 * m + rng.normal(0, 0.001, 400)})
    b = rolling_beta(r, m, window=252, min_periods=126)
    assert b["A"].iloc[-1] == pytest.approx(1.5, abs=0.05)
    assert b["B"].iloc[-1] == pytest.approx(-0.5, abs=0.05)
    assert b.iloc[:125].isna().all().all()


def test_universe_requires_history_and_liquidity(prices):
    vol = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
    vol["T01"] = 1.0  # illiquid
    wide = {"close": prices, "adj_close": prices, "volume": vol}
    ucfg = {"adv_window": 20, "min_adv_usd": 1e7, "min_history": 100}
    uni = build_universe(wide, None, ucfg)
    assert not uni["T01"].any()
    assert not uni.iloc[:100].any().any()  # first valid return is day 1 -> 100 returns at row 100
    assert uni.iloc[100:]["T00"].all()
