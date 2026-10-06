import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation import ic_summary, mean_cs_correlation, period_dates, rank_ic
from src.factors import momentum_raw, reversal_raw, volatility_raw
from src.preprocessing import daily_returns, forward_returns


def test_reversal_raw_is_past_return(prices):
    r5 = reversal_raw(prices, 5)
    assert r5.iloc[10]["T00"] == pytest.approx(prices.iloc[10]["T00"] / prices.iloc[5]["T00"] - 1)


def test_momentum_skips_recent_days(prices):
    m = momentum_raw(prices, 60, skip=5)
    i = 100
    assert m.iloc[i]["T02"] == pytest.approx(prices.iloc[i - 5]["T02"] / prices.iloc[i - 60]["T02"] - 1)


def test_volatility_window(prices):
    r = daily_returns(prices)
    v = volatility_raw(r, 20)
    i = 50
    expected = r.iloc[i - 19:i + 1]["T05"].std() * np.sqrt(252)
    assert v.iloc[i]["T05"] == pytest.approx(expected)


@pytest.mark.parametrize("builder", [
    lambda p, r: reversal_raw(p, 5),
    lambda p, r: momentum_raw(p, 60, 5),
    lambda p, r: volatility_raw(r, 20),
])
def test_factors_have_no_lookahead(prices, builder):
    """Changing prices after date t must not change any factor value at or before t."""
    t = 150
    f1 = builder(prices, daily_returns(prices))
    p2 = prices.copy()
    p2.iloc[t + 1:] *= np.random.default_rng(3).uniform(0.5, 2.0, size=p2.iloc[t + 1:].shape)
    f2 = builder(p2, daily_returns(p2))
    pd.testing.assert_frame_equal(f1.iloc[:t + 1], f2.iloc[:t + 1])


def test_rank_ic_matches_scipy_spearman(prices):
    rng = np.random.default_rng(4)
    sig = pd.DataFrame(rng.normal(size=prices.shape), index=prices.index, columns=prices.columns)
    sig.iloc[10, 3] = np.nan
    fwd = forward_returns(prices, 1)
    ic = rank_ic(sig, fwd)
    for i in (10, 77):
        s, f = sig.iloc[i], fwd.iloc[i]
        ok = s.notna() & f.notna()
        assert ic.iloc[i] == pytest.approx(stats.spearmanr(s[ok], f[ok]).statistic)


def test_rank_ic_perfect_signal(prices):
    fwd = forward_returns(prices, 5)
    ic = rank_ic(fwd * 2 + 1, fwd).dropna()
    assert np.allclose(ic, 1.0)
    assert np.allclose(rank_ic(-fwd, fwd).dropna(), -1.0)


def test_ic_summary_fields():
    ic = pd.Series([0.1, -0.05, 0.02, 0.03, 0.0, 0.04, 0.01, -0.02, 0.05, 0.02, 0.03])
    s = ic_summary(ic, 1)
    assert s["mean_ic"] == pytest.approx(ic.mean())
    assert s["ic_ir"] == pytest.approx(ic.mean() / ic.std())
    assert s["pct_positive"] == pytest.approx((ic > 0).mean())


def test_period_dates_purges_end():
    idx = pd.bdate_range("2020-01-01", periods=30)
    d = period_dates(idx, "2020-01-01", "2020-02-11", purge=5)
    full = idx[idx <= "2020-02-11"]
    assert len(d) == len(full) - 5
    assert d[-1] == full[-6]


def test_mean_cs_correlation_identity(prices):
    c = mean_cs_correlation({"a": prices, "b": prices * 2, "c": -prices}, prices.index)
    assert c.loc["a", "b"] == pytest.approx(1.0)
    assert c.loc["a", "c"] == pytest.approx(-1.0)
