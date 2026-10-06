import numpy as np
import pandas as pd
import pytest

from src.backtest import rebalance_schedule, run_backtest
from src.metrics import drawdown, information_ratio, market_beta, sharpe
from src.portfolio import ex_ante_beta, exposures, long_short_weights
from src.preprocessing import daily_returns


def test_weights_dollar_neutral(prices):
    rng = np.random.default_rng(5)
    sig = pd.DataFrame(rng.normal(size=prices.shape), index=prices.index, columns=prices.columns)
    for weighting in ("equal", "signal"):
        w = long_short_weights(sig, 0.2, weighting, gross=1.0)
        e = exposures(w)
        assert np.allclose(e["net_exposure"], 0, atol=1e-12)
        assert np.allclose(e["gross_exposure"], 1.0)
        assert (e["n_long"] == 8).all() and (e["n_short"] == 8).all()


def test_weights_long_top_short_bottom(prices):
    sig = pd.DataFrame(np.tile(np.arange(40.0), (len(prices), 1)), index=prices.index, columns=prices.columns)
    w = long_short_weights(sig, 0.25)
    assert (w.iloc[0, -10:] > 0).all() and (w.iloc[0, :10] < 0).all()
    assert (w.iloc[0, 10:30] == 0).all()


def test_weights_zero_when_too_few_names(prices):
    sig = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    sig.iloc[:, :6] = 1.0 + np.arange(6)
    w = long_short_weights(sig, 0.2, min_per_leg=5)
    assert (w == 0).all().all()


def test_ex_ante_beta():
    w = pd.DataFrame([[0.5, -0.5]], columns=["a", "b"])
    b = pd.DataFrame([[1.2, 0.8]], columns=["a", "b"])
    assert ex_ante_beta(w, b).iloc[0] == pytest.approx(0.2)


def _toy():
    idx = pd.bdate_range("2021-01-01", periods=6)
    r = pd.DataFrame({"A": [0, 0.10, 0.0, 0.05, 0, 0], "B": [0, -0.10, 0.0, 0.0, 0, 0]}, index=idx, dtype=float)
    w = pd.DataFrame({"A": [0.5] * 6, "B": [-0.5] * 6}, index=idx)
    return idx, r, w


def test_backtest_timing_lag0():
    idx, r, w = _toy()
    daily, _ = run_backtest(w, r, idx[:1], execution_lag=0, cost_bps=0, slippage_bps=0)
    # trade at close of day 0; first return earned is day 1: 0.5*0.10 + (-0.5)*(-0.10) = 0.10
    assert daily["gross_return"].iloc[1] == pytest.approx(0.10)
    assert daily["gross_return"].iloc[0] == pytest.approx(0.0)


def test_backtest_timing_lag1_misses_first_move():
    idx, r, w = _toy()
    daily, _ = run_backtest(w, r, idx[:1], execution_lag=1, cost_bps=0, slippage_bps=0)
    # signal at day 0, trade at close day 1 -> day-1 move is not captured
    assert daily.index[0] == idx[1]
    assert daily["gross_return"].iloc[0] == pytest.approx(0.0)


def test_backtest_costs_deducted_from_turnover():
    idx, r, w = _toy()
    daily, _ = run_backtest(w, r, idx[:1], execution_lag=0, cost_bps=3, slippage_bps=7)
    # initial build: turnover = gross = 1.0
    assert daily["turnover"].iloc[0] == pytest.approx(1.0)
    assert daily["transaction_cost"].iloc[0] == pytest.approx(3e-4)
    assert daily["slippage"].iloc[0] == pytest.approx(7e-4)
    assert np.allclose(daily["net_return"], daily["gross_return"] - daily["transaction_cost"] - daily["slippage"])


def test_backtest_drift_and_rebalance_turnover():
    idx, r, w = _toy()
    daily, hold = run_backtest(w, r, idx[[0, 2]], execution_lag=0, cost_bps=0, slippage_bps=0)
    # after day 1: A 0.5*1.1=0.55, B -0.5*0.9=-0.45, capital 1.10 -> weights 0.5, -0.409
    assert hold.iloc[1]["A"] == pytest.approx(0.55 / 1.1)
    assert hold.iloc[1]["B"] == pytest.approx(-0.45 / 1.1)
    expected_turnover = abs(0.5 - 0.55 / 1.1) + abs(-0.5 + 0.45 / 1.1)
    assert daily["turnover"].iloc[2] == pytest.approx(expected_turnover)
    # holdings are fixed between rebalances: no trade on day 3
    assert daily["turnover"].iloc[3] == 0


def test_backtest_no_lookahead(prices):
    """Weights at signal date s must not affect returns at or before s + lag."""
    r = daily_returns(prices)
    rng = np.random.default_rng(6)
    w = long_short_weights(pd.DataFrame(rng.normal(size=prices.shape), index=prices.index, columns=prices.columns))
    sched = rebalance_schedule(prices.index, None, 5)
    d1, _ = run_backtest(w, r, sched, execution_lag=1)
    w2 = w.copy()
    w2.iloc[200:] = -w2.iloc[200:]
    d2, _ = run_backtest(w2, r, sched, execution_lag=1)
    cut = prices.index[201]
    pd.testing.assert_series_equal(d1.loc[:cut, "gross_return"], d2.loc[:cut, "gross_return"])
    assert not np.allclose(d1.loc[prices.index[210]:, "gross_return"], d2.loc[prices.index[210]:, "gross_return"])


def test_backtest_neutral_book_stays_near_neutral(prices):
    r = daily_returns(prices)
    rng = np.random.default_rng(7)
    w = long_short_weights(pd.DataFrame(rng.normal(size=prices.shape), index=prices.index, columns=prices.columns))
    d, _ = run_backtest(w, r, rebalance_schedule(prices.index, None, 1), execution_lag=1)
    assert d["net_exposure"].abs().max() < 1e-12  # rebalanced daily: exactly neutral at each close


def test_sharpe_and_ir():
    r = pd.Series([0.01, -0.005, 0.002, 0.004])
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    b = pd.Series([0.005, 0.0, 0.001, 0.0])
    a = r - b
    assert information_ratio(r, b) == pytest.approx(a.mean() / a.std() * np.sqrt(252))


def test_drawdown():
    r = pd.Series([0.10, -0.50, 0.20])
    dd = drawdown(r)
    assert dd.iloc[0] == 0
    assert dd.iloc[1] == pytest.approx(-0.5)
    assert dd.iloc[2] == pytest.approx(1.1 * 0.5 * 1.2 / 1.1 - 1)
    # loss on day one counts from initial capital 1.0
    assert drawdown(pd.Series([-0.2])).iloc[0] == pytest.approx(-0.2)


def test_market_beta_regression():
    rng = np.random.default_rng(8)
    m = pd.Series(rng.normal(0, 0.01, 1000))
    r = 0.3 * m + rng.normal(0, 0.002, 1000)
    assert market_beta(r, m)["beta"] == pytest.approx(0.3, abs=0.02)
