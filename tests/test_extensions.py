import numpy as np
import pandas as pd
import pytest

from src.extensions import hysteresis_weights
from src.portfolio import exposures


def _ranked(values_by_day):
    idx = pd.bdate_range("2021-01-01", periods=len(values_by_day))
    return pd.DataFrame(values_by_day, index=idx, columns=[f"S{i}" for i in range(len(values_by_day[0]))])


def test_hysteresis_entry_and_exit():
    # 20 stocks; entry top/bottom 10% (2 names), exit outside top/bottom 30% (6 names)
    day0 = np.arange(20.0)                      # S18, S19 long; S0, S1 short
    day1 = day0.copy(); day1[[18, 19]] = [15, 14]  # S18/S19 fall to ranks 16th/15th: inside exit band
    day1[[16, 17]] = [18.5, 19.5]                # new top names enter
    day2 = day1.copy(); day2[19] = 5.0           # S19 drops out of the exit band
    sig = _ranked([day0, day1, day2])
    w = hysteresis_weights(sig, sig.index, entry_q=0.1, exit_q=0.3, min_per_leg=2)
    assert set(w.columns[w.iloc[0] > 0]) == {"S18", "S19"}
    assert set(w.columns[w.iloc[1] > 0]) == {"S16", "S17", "S18", "S19"}
    assert set(w.columns[w.iloc[2] > 0]) == {"S16", "S17", "S18"}
    assert set(w.columns[w.iloc[2] < 0]) == {"S0", "S1"}


def test_hysteresis_dollar_neutral_and_gross():
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.normal(size=(50, 100)), index=pd.bdate_range("2021-01-01", periods=50))
    w = hysteresis_weights(sig, sig.index, 0.1, 0.2)
    e = exposures(w)
    assert np.allclose(e["net_exposure"], 0, atol=1e-12)
    assert np.allclose(e["gross_exposure"], 1.0)
    assert (e["n_long"] >= 10).all()


def test_hysteresis_equals_quantile_book_when_bands_equal():
    from src.portfolio import long_short_weights
    rng = np.random.default_rng(1)
    sig = pd.DataFrame(rng.normal(size=(20, 60)), index=pd.bdate_range("2021-01-01", periods=20))
    pd.testing.assert_frame_equal(hysteresis_weights(sig, sig.index, 0.2, 0.2),
                                  long_short_weights(sig, 0.2), check_freq=False)


def test_hysteresis_reduces_turnover():
    rng = np.random.default_rng(2)
    base = rng.normal(size=100)
    days = [base + 0.5 * rng.normal(size=100) for _ in range(40)]  # persistent signal + noise
    sig = _ranked(days)
    tight = hysteresis_weights(sig, sig.index, 0.1, 0.1)
    buffered = hysteresis_weights(sig, sig.index, 0.1, 0.25)
    to = lambda w: w.diff().abs().sum(axis=1).iloc[1:].mean()  # noqa: E731
    assert to(buffered) < to(tight)


def test_hysteresis_rejects_bad_bands():
    sig = _ranked([np.arange(10.0)])
    with pytest.raises(ValueError):
        hysteresis_weights(sig, sig.index, 0.3, 0.2)
